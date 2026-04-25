from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.client_profile import ClientProfile, RiskTolerance


@dataclass
class AllocationTarget:
    equities_pct: float
    bonds_pct: float
    real_estate_pct: float
    cash_pct: float

    def __str__(self) -> str:
        return (
            f"Equities: {self.equities_pct:.0f}% | "
            f"Bonds: {self.bonds_pct:.0f}% | "
            f"Real Estate: {self.real_estate_pct:.0f}% | "
            f"Cash: {self.cash_pct:.0f}%"
        )


@dataclass
class StrategyRecommendation:
    strategy_name: str
    allocation: AllocationTarget
    rationale: str
    suggested_instruments: list[str]
    rebalancing_frequency: str
    key_risks: list[str]

    def to_prompt_context(self) -> str:
        instruments = ", ".join(self.suggested_instruments)
        risks = "; ".join(self.key_risks)
        return (
            f"Strategy: {self.strategy_name}\n"
            f"Target Allocation: {self.allocation}\n"
            f"Rationale: {self.rationale}\n"
            f"Suggested Instruments: {instruments}\n"
            f"Rebalancing: {self.rebalancing_frequency}\n"
            f"Key Risks: {risks}"
        )


class BaseInvestmentStrategy(ABC):
    """Abstract base for all investment strategies (Strategy Pattern)."""

    @abstractmethod
    def get_recommendation(self, profile: ClientProfile) -> StrategyRecommendation:
        pass


class ConservativeStrategy(BaseInvestmentStrategy):
    """Low-risk strategy focused on capital preservation and income."""

    def get_recommendation(self, profile: ClientProfile) -> StrategyRecommendation:
        return StrategyRecommendation(
            strategy_name="Conservative",
            allocation=AllocationTarget(
                equities_pct=25.0,
                bonds_pct=55.0,
                real_estate_pct=10.0,
                cash_pct=10.0,
            ),
            rationale=(
                f"{profile.name} has a conservative risk profile. "
                "The portfolio prioritizes capital preservation and steady income "
                "over growth, with a large bond allocation to reduce volatility."
            ),
            suggested_instruments=[
                "US Treasury Bonds (TLT)",
                "Investment-Grade Corporate Bond ETF (LQD)",
                "Dividend Equity ETF (VYM)",
                "High-Yield Savings / Money Market",
                "REIT Index ETF (VNQ)",
            ],
            rebalancing_frequency="Semi-annually",
            key_risks=[
                "Inflation erosion on bond-heavy portfolios",
                "Interest rate risk on long-duration bonds",
                "Opportunity cost vs. moderate/aggressive strategies",
            ],
        )


class ModerateStrategy(BaseInvestmentStrategy):
    """Balanced strategy targeting growth with managed volatility."""

    def get_recommendation(self, profile: ClientProfile) -> StrategyRecommendation:
        return StrategyRecommendation(
            strategy_name="Moderate (Balanced)",
            allocation=AllocationTarget(
                equities_pct=60.0,
                bonds_pct=30.0,
                real_estate_pct=7.0,
                cash_pct=3.0,
            ),
            rationale=(
                f"{profile.name} has a moderate risk tolerance with a "
                f"{profile.investment_horizon_years}-year horizon. "
                "A balanced 60/40 approach captures long-term equity growth "
                "while bonds provide a cushion during downturns."
            ),
            suggested_instruments=[
                "S&P 500 Index ETF (VOO)",
                "International Developed Market ETF (VXUS)",
                "Aggregate Bond ETF (BND)",
                "REIT Index ETF (VNQ)",
                "Short-Term Treasury ETF (VGSH)",
            ],
            rebalancing_frequency="Annually",
            key_risks=[
                "Equity market volatility",
                "Sequence of returns risk near retirement",
                "Currency risk on international allocation",
            ],
        )


class AggressiveStrategy(BaseInvestmentStrategy):
    """High-growth strategy accepting higher short-term volatility."""

    def get_recommendation(self, profile: ClientProfile) -> StrategyRecommendation:
        return StrategyRecommendation(
            strategy_name="Aggressive (Growth)",
            allocation=AllocationTarget(
                equities_pct=85.0,
                bonds_pct=10.0,
                real_estate_pct=5.0,
                cash_pct=0.0,
            ),
            rationale=(
                f"{profile.name} has an aggressive risk tolerance and a long "
                f"{profile.investment_horizon_years}-year horizon. "
                "A growth-oriented equity-heavy portfolio maximizes compounding "
                "potential over the long run."
            ),
            suggested_instruments=[
                "Total Stock Market ETF (VTI)",
                "Growth Stock ETF (VUG)",
                "Emerging Markets ETF (VWO)",
                "Small-Cap ETF (VB)",
                "High-Yield Corporate Bond ETF (HYG)",
            ],
            rebalancing_frequency="Annually",
            key_risks=[
                "High short-term drawdown potential (30-50% in bear markets)",
                "Emotional discipline required during corrections",
                "Concentration in equities increases volatility",
            ],
        )


class StrategyFactory:
    """Factory that maps RiskTolerance to the correct strategy implementation."""

    _registry: dict[RiskTolerance, BaseInvestmentStrategy] = {
        RiskTolerance.CONSERVATIVE: ConservativeStrategy(),
        RiskTolerance.MODERATE: ModerateStrategy(),
        RiskTolerance.AGGRESSIVE: AggressiveStrategy(),
    }

    @classmethod
    def get_strategy(cls, profile: ClientProfile) -> BaseInvestmentStrategy:
        strategy = cls._registry.get(profile.risk_tolerance)
        if strategy is None:
            raise ValueError(f"No strategy registered for risk tolerance: {profile.risk_tolerance}")
        return strategy
