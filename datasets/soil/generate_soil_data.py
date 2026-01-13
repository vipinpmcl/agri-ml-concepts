"""
Generate synthetic soil dataset for PCA learning
"""
import random
import csv

# Set random seed
random.seed(42)

# Number of samples
n_samples = 200

# Helper function for random uniform
def uniform(low, high):
    return low + random.random() * (high - low)

# Helper function for normal distribution
def normal(mean, std):
    # Box-Muller transform
    u1 = random.random()
    u2 = random.random()
    z0 = (-2 * (u1 if u1 > 1e-10 else 1e-10).__log__())**0.5 * (2 * 3.14159265359 * u2).__cos__()
    return mean + z0 * std

def normal_simple(mean, std):
    # Simpler approximation using CLT
    total = sum(random.random() for _ in range(12))
    return mean + std * (total - 6)

# Generate data
data = []

for i in range(1, n_samples + 1):
    sample_id = f'SOIL_{i:03d}'
    region = random.choice(['North', 'South', 'East', 'West', 'Central'])
    soil_type = random.choice(['Clay', 'Sandy', 'Loam', 'Silt', 'Clay-Loam'])

    # Generate correlated features
    pH = uniform(5.5, 8.5)
    organic_matter = uniform(1, 8)

    # Nitrogen correlated with organic matter
    nitrogen = 20 + organic_matter * 10 + normal_simple(0, 15)
    nitrogen = max(10, min(150, nitrogen))

    # Phosphorus correlated with nitrogen
    phosphorus = 15 + nitrogen * 0.3 + normal_simple(0, 10)
    phosphorus = max(5, min(100, phosphorus))

    # Potassium correlated with phosphorus
    potassium = 100 + phosphorus * 2 + normal_simple(0, 30)
    potassium = max(50, min(400, potassium))

    # Texture components
    sand = uniform(10, 70)
    clay = uniform(10, 60)
    total = sand + clay
    sand = sand / total * 90 + 5
    clay = clay / total * 90 + 5
    silt = 100 - sand - clay

    # Moisture
    moisture = uniform(10, 40)

    # Micronutrients
    iron = uniform(20, 200)
    zinc = uniform(1, 50)
    copper = uniform(0.5, 20)
    manganese = uniform(5, 100)
    boron = uniform(0.2, 5)

    # CEC correlated with clay and organic matter
    cec = 5 + clay * 0.3 + organic_matter * 2 + normal_simple(0, 3)
    cec = max(5, min(40, cec))

    # EC
    ec = uniform(0.1, 2.5)

    row = [
        sample_id, region, soil_type,
        round(pH, 2),
        round(organic_matter, 2),
        round(nitrogen, 1),
        round(phosphorus, 1),
        round(potassium, 1),
        round(sand, 1),
        round(silt, 1),
        round(clay, 1),
        round(moisture, 1),
        round(iron, 1),
        round(zinc, 2),
        round(copper, 2),
        round(manganese, 1),
        round(boron, 2),
        round(cec, 1),
        round(ec, 2)
    ]
    data.append(row)

# Write to CSV
output_path = '/home/vipin/1.EMastersAIML/11.AIMLProjectsWithRealWorldDatasets/workspace/datasets/soil/sample_soil_data.csv'

headers = [
    'sample_id', 'region', 'soil_type', 'pH', 'organic_matter_percent',
    'nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm',
    'sand_percent', 'silt_percent', 'clay_percent', 'moisture_percent',
    'iron_ppm', 'zinc_ppm', 'copper_ppm', 'manganese_ppm', 'boron_ppm',
    'cec_meq_100g', 'ec_ds_m'
]

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data)

print(f'Created soil dataset with {n_samples} samples')
print(f'Features: {len(headers)}')
print(f'Saved to: {output_path}')
print('\nFirst few rows:')
for i in range(min(5, len(data))):
    print(data[i][:6])  # Print first 6 columns
