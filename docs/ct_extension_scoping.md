# CT Quality Assessment Extension

## Objective

Extend the chest radiograph quality assessment framework to CT imaging.

## Potential Quality Dimensions

### Coverage

- Lung apex coverage
- Costophrenic angle coverage
- Complete thoracic coverage

### Motion

- Respiratory motion
- Patient movement

### Reconstruction

- Slice thickness
- Reconstruction kernel

### Technical

- Noise
- Contrast enhancement
- Artifact burden

## Proposed Pipeline

1. DICOM ingestion
2. Volume loading
3. Lung segmentation
4. Quality scoring modules
5. Composite score fusion
6. PDF reporting

## Expected Challenges

- Volumetric data
- Higher memory requirements
- Multiple acquisition protocols
- Larger annotation burden

## Recommendation

Use the existing modular scorer architecture as the base framework.