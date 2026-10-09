"""Exporta predicciones del modelo a replay.json para index.html.

Uso (al terminar un entrenamiento o al guardar un checkpoint):
    python export_replay.py --data mit-bih-arrhythmia-database-1.0.0 --records 220 --model Modelo2 --out .
    En caso de estar en otra carpeta que no sea root:
    python mi_carpeta/export_replay.py --data mit-bih-arrhythmia-database-1.0.0 --records 220 --model Modelo2 --out mi_carpeta.
    
El visor consulta replay.meta.json cada 10 s; si la versión cambia, recarga.
Hacer que se genere la animacion en el archivo html: 
    python -m http.server 800

    Exporta predicciones del modelo a replay.json para index.html.

Uso (al terminar un entrenamiento o al guardar un checkpoint):
    python export_replay.py --records 220 102 --model ckpt-s0 --out .

El visor consulta replay.meta.json cada 10 s; si la versión cambia, recarga.
"""
import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

FS, PRE, WIN, MAX_BEATS = 360, 144, 360, 25
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "A": "S", "a": "S", "J": "S",
        "S": "S", "V": "V", "E": "V", "F": "F", "/": "Q", "f": "Q", "Q": "Q"}
CLASSES = ["N", "S", "V", "F", "Q"]


def predict_windows(windows: np.ndarray):
    """Datos simulados temporales para ver la animación."""
    B = windows.shape[0]
    probs = np.random.rand(B, 5)
    probs /= probs.sum(axis=1, keepdims=True) # Probabilidades que suman 1
    recon = windows + 0.05                  # Reconstrucción simulada
    att = np.random.rand(B, 24)             # Atención simulada para los 24 nodos
    return probs, recon, att


def build_record(path: Path, rec_id: str):
    rec = wfdb.rdrecord(str(path / rec_id))
    ann = wfdb.rdann(str(path / rec_id), "atr")
    sig = rec.p_signal[:, 0].astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    n_seg = MAX_BEATS * FS
    idx = [(s, AAMI[y]) for s, y in zip(ann.sample, ann.symbol) if y in AAMI and s < n_seg + 5 * FS]
    idx = [(s, c) for s, c in idx if s - PRE >= 0 and s - PRE + WIN <= len(sig)][:MAX_BEATS]
    wins = np.stack([sig[s - PRE:s - PRE + WIN] for s, _ in idx])
    probs, recon, att = predict_windows(wins)
    end = idx[-1][0] + FS
    beats = []
    for k, (s, c) in enumerate(idx):
        p = int(probs[k].argmax())
        b = {"i": int(s), "true": c, "pred": CLASSES[p], "conf": float(probs[k][p]),
             "target": wins[k].round(3).tolist(), "recon": recon[k].round(3).tolist()}
        if att is not None:
            b["att"] = (att[k] / (att[k].max() + 1e-8)).round(3).tolist()
        beats.append(b)
    ok = sum(b["pred"] == b["true"] for b in beats)
    print(f"{rec_id}: exactitud {ok}/{len(beats)} | reales {Counter(b['true'] for b in beats)} | predichos {Counter(b['pred'] for b in beats)}")
    if ok / len(beats) < 0.7:
        print("  AVISO: exactitud baja; revisa orden de clases, model.eval(), alineacion R=144 y entradas RR del modelo")
    return {"id": rec_id, "signal": sig[:end].round(3).tolist(), "beats": beats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mit-bih-arrhythmia-database-1.0.0")
    ap.add_argument("--records", nargs="+", default=["220"])
    ap.add_argument("--model", default="modelo")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    version = time.strftime("%Y-%m-%d %H:%M:%S")
    data = {"version": version, "model": a.model, "fs": FS,
            "records": [build_record(Path(a.data), r) for r in a.records]}
    out = Path(a.out)
    (out / "replay.json").write_text(json.dumps(data, separators=(",", ":")))
    # El meta se escribe al final, para que el visor nunca lea un replay a medio escribir.
    (out / "replay.meta.json").write_text(json.dumps({"version": version}))
    print(f"replay.json actualizado ({version})")