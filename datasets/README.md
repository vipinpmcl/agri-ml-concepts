# Datasets

This directory contains agricultural datasets used throughout the learning modules.

## Directory Structure

```
datasets/
├── soil/           # Soil chemistry and properties data
├── crop/           # Crop yield and characteristics (future)
├── weather/        # Weather and climate data (future)
└── remote_sensing/ # Satellite imagery data (future)
```

## Available Datasets

### Soil Data

**Location**: `datasets/soil/`

Soil chemistry datasets containing various soil properties used in PCA analysis and other ML modules.

**Typical Features**:
- pH levels (4.0 - 9.0)
- Nitrogen content (mg/kg)
- Phosphorus content (mg/kg)
- Potassium content (mg/kg)
- Organic matter percentage
- Soil texture (sand, silt, clay percentages)
- Micronutrients (Fe, Zn, Cu, Mn)
- Electrical conductivity
- Moisture content
- Soil type/classification

## Data Sources

### Recommended Public Datasets

1. **UCI Machine Learning Repository**
   - Various agricultural and soil datasets
   - URL: https://archive.ics.uci.edu/ml/datasets.php

2. **Kaggle**
   - Agricultural datasets
   - Soil chemistry data
   - URL: https://www.kaggle.com/datasets

3. **USDA NRCS**
   - National Cooperative Soil Survey
   - Comprehensive soil data
   - URL: https://www.nrcs.usda.gov/

4. **FAO Soils Portal**
   - Global soil database
   - URL: http://www.fao.org/soils-portal/

5. **ISRIC World Soil Information**
   - Global soil property maps
   - URL: https://www.isric.org/

## Creating Synthetic Data

For learning purposes, some modules may use synthetic but realistic agricultural data. The generation scripts ensure:

- Realistic value ranges based on actual agricultural data
- Appropriate correlations between features
- Normal and non-normal distributions where appropriate
- Outliers and missing values for real-world scenarios

## Data Usage Guidelines

### Large Files

Large dataset files (>10MB) are listed in `.gitignore` and should not be committed to the repository. Instead:

1. Download datasets from the sources above
2. Place them in the appropriate subdirectory
3. Document the source and download instructions

### Small Sample Files

Small example files (`sample_*.csv`) are included in the repository for:
- Quick testing
- Demonstration purposes
- Running notebooks without downloads

## Data Preparation

Before using datasets in notebooks:

1. **Load the data**: Use pandas to read CSV/Excel files
2. **Explore**: Check for missing values, outliers, data types
3. **Clean**: Handle missing values and outliers appropriately
4. **Standardize**: Scale features when needed for ML algorithms
5. **Split**: Divide into training and testing sets if applicable

## Adding New Datasets

When adding new datasets to the repository:

1. Create a subdirectory if needed
2. Add a description in this README
3. Include the data source and licensing information
4. Update `.gitignore` if the file is large
5. Provide a sample or instructions to download

## License and Attribution

When using public datasets:

- Always check the dataset license
- Provide proper attribution
- Include citations in your notebooks
- Respect usage restrictions

## Contact

For questions about datasets or to suggest new agricultural data sources, please open an issue in the repository.
