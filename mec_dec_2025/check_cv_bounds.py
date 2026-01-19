"""Quick script to check CloudVolume bounds."""
import cloudvolume as cv

# Check input volume
print("Checking input CloudVolume bounds...")
input_vol = cv.CloudVolume(
    "gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/",
    mip=0,
    use_https=False
)

print(f"Bounds: {input_vol.bounds}")
print(f"Shape: {input_vol.bounds.size3()}")
print(f"X: [{input_vol.bounds.minpt[0]}, {input_vol.bounds.maxpt[0]})")
print(f"Y: [{input_vol.bounds.minpt[1]}, {input_vol.bounds.maxpt[1]})")
print(f"Z: [{input_vol.bounds.minpt[2]}, {input_vol.bounds.maxpt[2]})")
print(f"Resolution: {input_vol.resolution}")
