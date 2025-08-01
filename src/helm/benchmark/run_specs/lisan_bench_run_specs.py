"""RunSpec registration for LisanBench.

This makes the description string ``lisan_bench`` usable in run-entries
files (HOCON/TOML).  HELM will expand the string into the RunSpec
returned below.
"""

from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.scenarios.lisan_bench import LisanBenchScenario


@run_spec_function("lisan_bench")
def get_lisan_bench_spec() -> RunSpec:  # noqa: D401
    """Return a minimal RunSpec wired to ``LisanBenchScenario``."""

    # The Scenario already defines a suitable adapter (prompt format),
    # so we instantiate it just to fetch the AdapterSpec.
    scenario = LisanBenchScenario()

    return RunSpec(
        name="lisan_bench",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.lisan_bench.LisanBenchScenario",
            args={},
        ),
        adapter_spec=scenario.get_adapter_spec(),
        metric_specs=get_basic_metric_specs(),
        groups=["lisan_bench"],
    )
