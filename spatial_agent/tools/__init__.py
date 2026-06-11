"""ToolsModule: injected into the Jupyter kernel as ``tools``.

Usage in kernel::

    recon = tools.Reconstruct.Reconstruct(InputImages[:32])
    bev = recon.render_bev(masks=seg)
    show(bev)
    seg = tools.SAM3.segment_image_by_text(InputImages[0], "car")
    mask_vis = seg.visualize(seg.frame_indices[0])
    tools.Time.frame_to_seconds(42)
    tools.Mask.centroid(seg.masks[0, 0])
    tools.Geometry.euclidean_distance(p1, p2)
    chart = tools.Graph.plot(distances, x_label="Frame", y_label="Dist (m)")
"""

from spatial_agent.tools.time_utils import TimeUtils
from spatial_agent.tools.mask_utils import MaskUtils
from spatial_agent.tools.geometry_utils import GeometryUtils
from spatial_agent.tools.graph_drawer import GraphDrawer
from spatial_agent.tools.draw_utils import DrawUtils


class ToolsModule:
    """Assembled tool namespace injected as ``tools`` in the kernel."""

    TOOL_NAMES = ("Reconstruct", "SAM3", "Graph", "Time", "Mask", "Geometry", "Draw")

    def __init__(
        self,
        config,
        metadata,
        input_images=None,
        input_images_list=None,
        video_sources_list=None,
    ):
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool

        gpu_retries = getattr(config, "gpu_tool_max_retries", 3)
        # SAM3 must be created before Reconstruct so the instance is available
        self.SAM3 = SAM3Tool(
            video_source=getattr(metadata, "video_source", None),
            input_images=input_images,
            gpu_tool_max_retries=gpu_retries,
            sam3_max_video_frames=getattr(config, "sam3_max_video_frames", 200),
            total_video_frames=metadata.total_frames or 0,
            num_videos=getattr(metadata, "num_videos", 1),
            input_images_list=input_images_list,
            video_sources_list=video_sources_list,
        )
        self.Reconstruct = ReconstructTool(
            self.SAM3, config,
            gpu_tool_max_retries=gpu_retries,
            metadata=metadata,
        )

        # CPU tools (direct, in-process)
        self.Graph = GraphDrawer()
        self.Time = TimeUtils(
            fps=metadata.fps or 1.0,
            total_frames=metadata.total_frames or 0,
        )
        self.Mask = MaskUtils()
        self.Geometry = GeometryUtils()
        self.Draw = DrawUtils()

    def get_all_prompt_descriptions(self, **format_kwargs) -> str:
        """Aggregate TOOL_PROMPT_DESCRIPTION from all tool instances.

        Args:
            **format_kwargs: Passed to str.format() on each description
                (e.g. reconstruct_max_frames=32).

        Returns:
            Concatenated tool descriptions ready for the system prompt.
        """
        from spatial_agent.config import get_config
        ablations = get_config().prompt_section_ablations

        tools = [
            self.Reconstruct, self.SAM3,
            self.Graph, self.Time, self.Mask, self.Geometry, self.Draw,
        ]
        parts = []
        for tool in tools:
            desc = tool.get_prompt_description(ablations=ablations)
            if desc:
                try:
                    desc = desc.format(**format_kwargs)
                except KeyError:
                    pass  # leave unformatted placeholders as-is
                parts.append(desc)
        return "\n".join(parts)

    @staticmethod
    def get_all_prompt_descriptions_static(**format_kwargs) -> str:
        """Aggregate descriptions from all tool classes (no instances needed).

        Only includes GPU tools that are listed in ``tools_to_use`` config.
        CPU tools are always included.  Ablation config is passed through so
        tool sub-sections can be individually excluded or overridden.
        """
        from spatial_agent.config import get_config
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool

        config = get_config()
        tools_to_use = set(config.tools_to_use or [])
        ablations = config.prompt_section_ablations

        # GPU tools: only include if configured
        gpu_tool_map = {
            "Reconstruct": ReconstructTool,
            "SAM3": SAM3Tool,
        }
        # CPU tools: always included
        cpu_classes = [GraphDrawer, TimeUtils, MaskUtils, GeometryUtils, DrawUtils]

        parts = []
        for name, cls in gpu_tool_map.items():
            if name in tools_to_use:
                desc = cls.get_prompt_description(ablations=ablations)
                if desc:
                    try:
                        desc = desc.format(**format_kwargs)
                    except KeyError:
                        pass
                    parts.append(desc)

        for cls in cpu_classes:
            desc = cls.get_prompt_description(ablations=ablations)
            if desc:
                try:
                    desc = desc.format(**format_kwargs)
                except KeyError:
                    pass
                parts.append(desc)

        return "\n".join(parts)

    @staticmethod
    def get_all_tool_section_names() -> set:
        """Collect all valid tool ablation section names for typo warnings."""
        from spatial_agent.tools.base import get_all_tool_ablation_names
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool

        return get_all_tool_ablation_names(
            ReconstructTool, SAM3Tool,
            GraphDrawer, TimeUtils, MaskUtils, GeometryUtils, DrawUtils,
        )

    def __repr__(self) -> str:
        tools = self.TOOL_NAMES
        return f"ToolsModule(available={tools})"
