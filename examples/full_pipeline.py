"""Full pipeline example: ingest → vectors → decisions → exports."""

import os
import tempfile
import openmind

# Create a sample repo for the demo
repo_dir = tempfile.mkdtemp(prefix="openmind-demo-")

# Write sample code
(repo_dir / "__init__.py").write_text("")
(repo_dir / "auth.py").write_text('''
def login(username: str, password: str) -> str:
    """Authenticate user and return token."""
    if check_credentials(username, password):
        return create_token(username)
    raise PermissionError("Invalid credentials")

def check_credentials(username: str, password: str) -> bool:
    """Check if credentials are valid."""
    return len(username) > 0 and len(password) > 0

def create_token(user_id: str) -> str:
    """Create JWT-like token."""
    return f"token_{user_id}"

def logout(token: str) -> bool:
    """Invalidate a token."""
    return True
''')

print("=" * 60)
print("OpenMind Full Pipeline Demo")
print("=" * 60)

# Step 1: Ingest
print("\n[1] Ingesting...")
result = openmind.ingest_repo(repo_dir)
print(f"    Functions: {len(result.functions)}")
print(f"    Classes: {len(result.classes)}")
print(f"    Stats: {result.stats}")

# Step 2: Build vectors
print("\n[2] Building vectors...")
builder = openmind.VectorBuilder(db_path=os.path.join(repo_dir, "vectors.db"))
vectors = builder.build_all(result)
print(f"    Vectors built: {len(vectors)}")

# Step 3: Search
print("\n[3] Searching for 'authentication'...")
matches = builder.search_input("authentication")
for dv, score in matches[:3]:
    print(f"    {dv.function_name}: {score:.4f}")

# Step 4: Tripartite decisions
print("\n[4] Making tripartite decisions...")
sync = openmind.TripartiteSynchronizer()
hw = openmind.TriHardwareProfile(gpu_available=False, device_type="desktop")
for func in result.functions:
    app = openmind.TriApplicationProfile(
        latency_requirement_ms=100,
        deterministic=True,
    )
    user = openmind.TriUserProfile(wants_manual_control=True)
    decision = sync.decide(hw, app, user)
    print(f"    {func.name}: {decision.value}")

# Step 5: Export
print("\n[5] Exporting...")
lever_record = openmind.export_lever_pack(
    function_name="login",
    module_path="auth",
    description="Authenticate user and return token",
)
print(f"    Lever record: {lever_record}")

nail_manifest = openmind.export_nail(
    function_name="check_credentials",
    cached_output={"valid": True},
    description="Pre-computed credential check",
    export_dir=os.path.join(repo_dir, "nails"),
)
print(f"    Nail manifest: {nail_manifest['function']}")

print("\nDone!")

# Cleanup
import shutil
shutil.rmtree(repo_dir, ignore_errors=True)
