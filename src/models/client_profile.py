from enum import Enum
from pydantic import BaseModel, Field


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentGoal(str, Enum):
    RETIREMENT = "retirement"
    WEALTH_GROWTH = "wealth_growth"
    INCOME_GENERATION = "income_generation"
    CAPITAL_PRESERVATION = "capital_preservation"
    EDUCATION_FUND = "education_fund"


class Holding(BaseModel):
    asset_name: str = Field(..., description="Name of the asset or fund")
    asset_type: str = Field(..., description="e.g. stock, ETF, bond, real_estate, cash")
    value_usd: float = Field(..., ge=0, description="Current market value in USD")
    allocation_pct: float = Field(..., ge=0, le=100, description="Percentage of portfolio")


class ClientProfile(BaseModel):
    name: str = Field(..., description="Client full name")
    age: int = Field(..., ge=18, le=100, description="Client age")
    annual_income_usd: float = Field(..., ge=0, description="Annual gross income in USD")
    total_assets_usd: float = Field(..., ge=0, description="Total net assets in USD")
    total_liabilities_usd: float = Field(default=0.0, ge=0, description="Total liabilities in USD")
    risk_tolerance: RiskTolerance = Field(..., description="Client risk tolerance level")
    investment_goals: list[InvestmentGoal] = Field(
        ..., min_length=1, description="Client investment objectives"
    )
    investment_horizon_years: int = Field(
        ..., ge=1, le=50, description="Investment time horizon in years"
    )
    current_holdings: list[Holding] = Field(
        default_factory=list, description="Current investment portfolio"
    )
    additional_notes: str = Field(
        default="", description="Any extra context about the client"
    )

    @property
    def net_worth_usd(self) -> float:
        return self.total_assets_usd - self.total_liabilities_usd

    def to_summary(self) -> str:
        goals = ", ".join(g.value for g in self.investment_goals)
        holdings_summary = (
            ", ".join(f"{h.asset_name} ({h.allocation_pct:.0f}%)" for h in self.current_holdings)
            if self.current_holdings
            else "None"
        )
        return (
            f"Client: {self.name}, Age: {self.age}\n"
            f"Annual Income: ${self.annual_income_usd:,.0f} | "
            f"Net Worth: ${self.net_worth_usd:,.0f}\n"
            f"Risk Tolerance: {self.risk_tolerance.value} | "
            f"Horizon: {self.investment_horizon_years} years\n"
            f"Goals: {goals}\n"
            f"Current Holdings: {holdings_summary}"
        )


# Dummy client profile for demo purposes
DEMO_CLIENT = ClientProfile(
    name="Sarah Chen",
    age=38,
    annual_income_usd=185_000,
    total_assets_usd=620_000,
    total_liabilities_usd=180_000,
    risk_tolerance=RiskTolerance.MODERATE,
    investment_goals=[InvestmentGoal.RETIREMENT, InvestmentGoal.WEALTH_GROWTH],
    investment_horizon_years=25,
    current_holdings=[
        Holding(asset_name="S&P 500 Index Fund", asset_type="ETF", value_usd=180_000, allocation_pct=40.0),
        Holding(asset_name="US Treasury Bonds", asset_type="bond", value_usd=90_000, allocation_pct=20.0),
        Holding(asset_name="Apple Inc.", asset_type="stock", value_usd=67_500, allocation_pct=15.0),
        Holding(asset_name="Real Estate (Rental)", asset_type="real_estate", value_usd=112_500, allocation_pct=25.0),
    ],
    additional_notes="Sarah is a senior software engineer. She is concerned about market volatility and prefers a balanced approach. She has a mortgage on her primary residence.",
)
