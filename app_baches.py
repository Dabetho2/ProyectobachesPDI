import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from ultralytics import YOLO
import numpy as np

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

MODEL_PATH = r"E:\UNIVERSIDAD\PDI\Proyectobaches\runs_detect\potholes_finalRTX\weights\best.pt"
OUTPUT_DIR = r"E:\UNIVERSIDAD\PDI\Proyectobaches\Resultados_Finales"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# ============================================================
# FUNCIÓN PARA CLASIFICAR SEVERIDAD
# ============================================================

def calcular_severidad(x1, y1, x2, y2, img_w, img_h):
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    ratio = box_area / img_area

    if ratio < 0.01:
        return "Leve"
    elif ratio < 0.03:
        return "Moderado"
    else:
        return "Severo"

# ============================================================
# FUNCIÓN PARA PROCESAR VIDEO
# ============================================================

def procesar_video(video_path):
    if not os.path.isfile(video_path):
        messagebox.showerror("Error", "El archivo no existe.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.basename(video_path).split(".")[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{filename}_detectado.mp4")

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    total_baches = 0
    severidad_contador = {"Leve": 0, "Moderado": 0, "Severo": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=960,
            verbose=False
        )

        res = results[0]
        annotated = frame.copy()

        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            # ======== CLASIFICAR SEVERIDAD =========
            sev = calcular_severidad(x1, y1, x2, y2, width, height)

            # Contador
            total_baches += 1
            severidad_contador[sev] += 1

            # Dibujar rectángulo
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 50, 50), 2)

            # Etiqueta
            cv2.putText(
                annotated,
                f"{sev} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 50, 50),
                2
            )

        # Mostrar en ventana
        cv2.imshow("Detecciones", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        out.write(annotated)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ==== MENSAJE FINAL =====
    msg = (
        f"✔ Procesamiento completado\n\n"
        f"Total de baches detectados: {total_baches}\n"
        f"Leves: {severidad_contador['Leve']}\n"
        f"Moderados: {severidad_contador['Moderado']}\n"
        f"Severos: {severidad_contador['Severo']}\n\n"
        f"Video guardado en:\n{output_video_path}"
    )
    messagebox.showinfo("Resultados", msg)


# ============================================================
# INTERFAZ TKINTER
# ============================================================

def seleccionar_video():
    ruta = filedialog.askopenfilename(
        title="Seleccionar video",
        filetypes=[("Videos", "*.mp4;*.avi;*.mov")]
    )
    if ruta:
        procesar_video(ruta)

ventana = tk.Tk()
ventana.title("Pothole Vision AI - Proyecto PDI David Thorp")
ventana.geometry("420x240")
ventana.resizable(False, False)

tk.Label(
    ventana, 
    text="Detector de Baches con YOLOv8\nPothole Vision AI",
    font=("Arial", 16, "bold")
).pack(pady=20)

tk.Button(
    ventana,
    text="Cargar Video",
    font=("Arial", 14),
    width=18,
    command=seleccionar_video
).pack()

tk.Label(
    ventana,
    text="Presiona 'q' para cerrar la vista del video.",
    fg="gray"
).pack(pady=15)

ventana.mainloop()
