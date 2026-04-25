from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    query: str = Field(..., description="The research question for the analyst")
    context: str = Field(
        ..., description="Background context from the advisor about why this is needed"
    )


class ResearchReport(BaseModel):
    task: ResearchTask = Field(..., description="The original research task")
    findings: str = Field(..., description="Analyst's synthesized research findings")
    sources: list[str] = Field(
        default_factory=list, description="List of sources or references used"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Analyst confidence score (0-1)"
    )
