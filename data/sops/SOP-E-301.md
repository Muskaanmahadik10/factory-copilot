# SOP-E-301: Position sensor fault: intermittent position error

## 1 Symptoms

Position error shows sudden spikes well above 10 um that appear and disappear between cycles; other sensors normal; no vibration or temperature change.

## 2 Diagnosis

Intermittent spikes without a mechanical signature point to the linear scale reader head, encoder cable or connector. A slow steady increase is calibration drift instead (see SOP-E-302).

## 3 Immediate action

Hold the axis. Inspect reader head for contamination and the encoder cable for damage.

## 4 Repair

Clean or replace the reader head; replace a chafed cable. Run the axis test program and confirm position error below 5 um with no spikes.
