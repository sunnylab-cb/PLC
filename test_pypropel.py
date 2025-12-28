import pypropel.fpseq as fpseq
import numpy as np

seq = "ACDEFGHIKLMNPQRSTVWY" # All 20 AAs

print("--- AAC ---")
aac = fpseq.composition(seq=seq, mode='aac')
print(f"Type: {type(aac)}")
print(f"Keys/Len: {len(aac)}")
first_val = list(aac.values())[0]
print(f"First Val: {first_val}")

print("\n--- DAC ---")
dac = fpseq.composition(seq=seq, mode='dac')
print(f"Type: {type(dac)}")
# Check if dict or list
if isinstance(dac, dict):
    print(f"Keys/Len: {len(dac)}")
    first_item = list(dac.values())[0]
    print(f"First Item: {first_item}")
elif isinstance(dac, list):
    print(f"Len: {len(dac)}")
    print(f"First Item: {dac[0]}")

print("\n--- AVEANF ---")
aveanf = fpseq.composition(seq=seq, mode='aveanf')
print(f"Type: {type(aveanf)}")
if isinstance(aveanf, dict):
    print(f"Keys/Len: {len(aveanf)}")
    print(f"Values: {list(aveanf.values())}")
