"""An LLM reads one recovery decision through the MCP server and returns a
validated diagnosis.

The causes and interventions are the repo's own vocabulary: core.friction.Driver
and recovery.playbook.Kind. Anything outside them fails validation.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Cause = Literal["price", "hesitation", "churn", "intent", "unknown"]
Intervention = Literal["discount", "reminder", "simplify", "none"]


class SessionDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_of_abandonment: Cause
    recommended_intervention: Intervention
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def no_offer_without_intent(self):
        if self.cause_of_abandonment == "intent" and self.recommended_intervention != "none":
            raise ValueError("a shopper who never intended to buy cannot be recovered; "
                             "recommended_intervention must be 'none'")
        return self


FALLBACK = SessionDiagnosis(
    cause_of_abandonment="unknown",
    recommended_intervention="none",
    confidence=0.0,
    rationale="fallback: no validated model output, so no action is recommended",
)
