"""ReturnAnswer class injected into the Jupyter kernel.

On valid answer, stores result in a builtins sentinel variable that the
feedback_node detects to terminate the agent loop.
"""

SENTINEL_NAME = "_return_answer_result"


class ReturnAnswer:
    """Submit the final answer and terminate the agent.

    Usage::

        ReturnAnswer("B")                              # multiple-choice letter
        ReturnAnswer(3)                                 # integer
        ReturnAnswer(3.14)                              # float
        ReturnAnswer("The dog is left of the cat")      # free-form text

    Accepts str, int, or float. The answer is stored as-is and also
    converted to a string representation for evaluation.
    """

    def __init__(self, answer):
        if not isinstance(answer, (str, int, float)):
            raise TypeError(
                f"ReturnAnswer accepts str, int, or float, got {type(answer).__name__}."
            )

        if isinstance(answer, str):
            answer = answer.strip()
            if not answer:
                raise ValueError("Answer must be a non-empty string.")

        import builtins

        result = {
            "text": str(answer),
            "raw_value": answer,
        }
        setattr(builtins, SENTINEL_NAME, result)
        self._result = result
        print(f"[ReturnAnswer] Answer submitted: {answer}")

    def __repr__(self) -> str:
        return f"ReturnAnswer(text='{self._result['text']}')"
