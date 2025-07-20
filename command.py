import subprocess

print("Generating Data")
subprocess.run(["python", "generate.py"], check=True)

print("Running OpenAI Experiments")
subprocess.run(["python", "zero-shot.py", "--models", "gpt-4o-mini"], check=True)
subprocess.run(["python", "zero-shot-CoT.py", "--models", "gpt-4o-mini"], check=True)
subprocess.run(["python", "mcq-zero-shot.py", "--models", "gpt-4o-mini"], check=True)
subprocess.run(["python", "mcq-zero-shot-CoT.py", "--models", "gpt-4o-mini"], check=True)
subprocess.run(["python", "few-shot.py", "--models", "gpt-4o-mini", "--num_shots", "1,2,3,4,5,6"], check=True)
