"""Basic usage example for openmind."""

import openmind

# Check version
print(f"OpenMind v{openmind.__version__}")

# Ingest a local repo
result = openmind.ingest_repo(".")

print(f"Functions: {len(result.functions)}")
print(f"Classes: {len(result.classes)}")
print(f"Test files: {len(result.test_files)}")
print(f"Call graph nodes: {len(result.call_graph)}")

# Show top functions
for func in sorted(result.functions, key=lambda f: len(f.calls), reverse=True)[:5]:
    print(f"  {func.signature} (calls: {len(func.calls)}, tested: {func.has_tests})")

# Build vectors
builder = openmind.VectorBuilder()
vectors = builder.build_all(result)
print(f"\nBuilt {len(vectors)} dual vectors")

# Search
matches = builder.search_input("parse")
print(f"\nSearch results for 'parse':")
for dv, score in matches:
    print(f"  {dv.function_name} ({dv.module}): {score:.4f}")

# Tripartite decision
sync = openmind.TripartiteSynchronizer()
hw = openmind.TriHardwareProfile()
app = openmind.TriApplicationProfile()
user = openmind.TriUserProfile()
decision = sync.decide(hw, app, user)
print(f"\nDefault decision: {decision.value}")

# Hardware probe
hw_caps = openmind.probe_hardware()
print(f"\nHardware: {hw_caps.device_type}, RAM={hw_caps.ram_gb}GB, GPU={hw_caps.gpu}")
