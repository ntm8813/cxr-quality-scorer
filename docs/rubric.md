# CXR 7-Axis Quality Rubric

This rubric defines the parameters for assigning `acceptable`, `borderline`, or `repeat` flags to chest radiographs across 7 distinct axes.

## 1. Sharpness
Evaluates motion blur and focus (measured via Laplacian variance).
* **Acceptable (1):** Crisp vascular markings, sharp diaphragm edges. Variance > 80.
* **Borderline (2):** Slight blurring, but clinically diagnostic. Variance 50-80.
* **Repeat (3):** Severe motion artifact; lung parenchyma obscured. Variance < 50.

## 2. Exposure
Evaluates penetration and contrast (measured via Exposure Index/Deviation Index).
* **Acceptable (1):** Clear visualization of retrocardiac spine and bronchovascular structures. DI between -1.0 and +1.0.
* **Borderline (2):** Mild over/under-penetration, but major structures visible. DI between -2.0 to -1.0 or +1.0 to +2.0.
* **Repeat (3):** Complete "white out" (under) or "burn out" (over). DI < -2.0 or > +2.0.

## 3. Rotation
Evaluates patient positioning (measured via clavicle landmark symmetry).
* **Acceptable (1):** Spinous processes centered between medial ends of clavicles. < 3° rotation.
* **Borderline (2):** Mild asymmetry, but mediastinal contours assessable. 3° - 5° rotation.
* **Repeat (3):** Severe rotation distorting heart size/mediastinum. > 5° rotation.

## 4. Coverage
Evaluates inclusion of necessary anatomical landmarks.
* **Acceptable (1):** Apices, costophrenic angles, and lateral rib margins fully included. Margin > 10px.
* **Borderline (2):** Minimal clipping of extreme apices or deep sulci, not affecting diagnosis. Margin 0-10px.
* **Repeat (3):** Major clipping (e.g., entire costophrenic angle missing). Lung mask touches image edge.

## 5. Inspiration
Evaluates lung expansion.
* **Acceptable (1):** 9-10 posterior ribs visible above the diaphragm.
* **Borderline (2):** 7-8 posterior ribs visible.
* **Repeat (3):** < 7 posterior ribs visible; crowding of lung markings mimicking disease.

## 6. Artifact
Evaluates foreign objects or technical artifacts.
* **Acceptable (1):** No artifacts, or known external medical devices (e.g., pacemaker, lines) that do not obscure pathology.
* **Borderline (2):** Removable artifacts (e.g., clothing snaps, ECG leads) partially obscuring lung fields.
* **Repeat (3):** Severe artifacts (e.g., grid lines, large foreign bodies) destroying diagnostic value.

## 7. Metadata
Evaluates DICOM header integrity.
* **Acceptable (1):** All required tags present and valid.
* **Borderline (2):** Non-critical tags missing (e.g., Patient Weight), but UID and Modality intact.
* **Repeat (3):** Critical routing/identification tags missing or corrupted.