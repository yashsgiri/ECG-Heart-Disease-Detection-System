import os
import threading
import time
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False
try:
    import serial
    SERIAL_AVAILABLE = True
except:
    SERIAL_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


class ECGApp:
    def __init__(self, master):
        self.master = master
        master.title("ECG Analysis System (Auto-Label ML)")
        master.geometry("520x460")
        self.time = np.array([])
        self.raw = np.array([])
        self.filtered = np.array([])
        self.fs = 250
        self.serial_conn = None
        self.live_mode = False
        self.live_buffer = []
        self.read_thread = None
        self.model = None
        title = tk.Label(master, text="ECG Analysis System", font=("Helvetica", 16))
        title.pack(pady=(12,8))
        btn_w = 30
        pady = 6
        tk.Button(master, text="Load CSV", width=btn_w, command=self.load_csv).pack(pady=pady)
        tk.Button(master, text="Analyze Loaded Data", width=btn_w, command=self.analyze_loaded).pack(pady=pady)
        tk.Button(master, text="Start Live Read (Arduino)", width=btn_w, command=self.start_live_read).pack(pady=pady)
        tk.Button(master, text="Stop Live Read", width=btn_w, command=self.stop_live_read).pack(pady=pady)
        tk.Button(master, text="Save Results to Excel", width=btn_w, command=self.save_excel).pack(pady=pady)
        tk.Button(master, text="Generate PDF Report", width=btn_w, command=self.generate_pdf).pack(pady=pady)
        tk.Label(master,text="").pack()
        tk.Button(master, text="Train ML Model (Auto-label)", width=40, command=self.train_ml).pack(pady=4)
        tk.Button(master, text="Predict with ML Model", width=40, command=self.predict_ml).pack(pady=4)
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Idle")
        status = tk.Label(master, textvariable=self.status_var, font=("Arial", 11))
        status.pack(side=tk.BOTTOM, pady=10)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9,4))


    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, order=3):
        if len(signal)<30: return signal
        ny=0.5*self.fs
        b,a=butter(order,[lowcut/ny,highcut/ny],btype='band')
        try: return filtfilt(b,a,signal)
        except: return signal


    def detect_rpeaks(self, sig):
        dist=int(0.25*self.fs)
        prom=max(0.5*np.std(sig),1e-3)
        peaks,_=find_peaks(sig,distance=dist,prominence=prom)
        return peaks


    def compute_metrics(self, data):
        filt=self.bandpass_filter(data)
        peaks=self.detect_rpeaks(filt)
        if len(peaks)>=2:
            rr=np.diff(peaks)/self.fs
            bpm=60.0/np.mean(rr)
            sdnn=np.std(rr)
            rmssd=np.sqrt(np.mean(np.diff(rr)**2))
        else:
            bpm=sdnn=rmssd=0.0
        return np.mean(filt), np.std(filt), len(peaks), bpm, sdnn, rmssd, peaks, filt


    def auto_label(self, bpm, sdnn, rmssd):
        if sdnn > 0.10 or rmssd > 0.08:
            return "Arrhythmia (High)"
        elif sdnn > 0.07 or rmssd > 0.06:
            return "Arrhythmia (Medium)"
        elif sdnn > 0.05 or rmssd > 0.04:
            return "Arrhythmia (Low)"

        if bpm < 40:
            return "Bradycardia (High)"
        elif bpm < 50:
            return "Bradycardia (Medium)"
        elif bpm < 60:
            return "Bradycardia (Low)"

        if bpm > 150:
            return "Tachycardia (High)"
        elif bpm > 120:
            return "Tachycardia (Medium)"
        elif bpm > 100:
            return "Tachycardia (Low)"

        return "Normal"


    def load_csv(self):
        path=filedialog.askopenfilename()
        if not path: return
        df=pd.read_csv(path)
        if 'timestamp' not in df.columns or 'ecg_value' not in df.columns:
            messagebox.showerror("Format Error","CSV must have timestamp & ecg_value")
            return
        self.time=df['timestamp'].values
        self.raw=df['ecg_value'].values.astype(float)
        if len(self.time)>1:
            self.fs=int(round(1/np.mean(np.diff(self.time))))
        else:
            self.fs=250
        self.status_var.set(f"Loaded {len(self.raw)} samples @ {self.fs}Hz")


    def analyze_loaded(self):
        if len(self.raw)==0:
            messagebox.showerror("No Data","Load CSV first")
            return
        meanv,stdv,pc,bpm,sdnn,rmssd,peaks,filt=self.compute_metrics(self.raw)
        t=np.linspace(0,len(filt)/self.fs,len(filt))
        self.ax.clear(); self.ax.plot(t,filt)
        if len(peaks)>0: self.ax.scatter(peaks/self.fs,filt[peaks],color='red')
        self.ax.set_title(f"Offline HR={bpm:.1f}")
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        
        label = self.auto_label(bpm, sdnn, rmssd)
        # Severity popup alert example
        if "High" in label:
            messagebox.showwarning("Severity Alert", f"High severity detected:\n{label}")
        elif "Medium" in label:
            messagebox.showinfo("Severity Notice", f"Medium severity detected:\n{label}")
        
        self.status_var.set(f"Offline HR={bpm:.1f} | {label}")
        messagebox.showinfo("Analysis",f"HR={bpm:.1f}\nSDNN={sdnn:.4f}\nRMSSD={rmssd:.4f}\nLabel={label}")


    def start_live_read(self):
        if not SERIAL_AVAILABLE:
            messagebox.showerror("Missing","pyserial not installed")
            return
        port=simpledialog.askstring("COM","Enter COM port (e.g. COM3):")
        if not port: return
        try:
            self.serial_conn=serial.Serial(port,9600,timeout=1)
        except Exception as e:
            messagebox.showerror("Error",str(e)); return
        self.live_buffer=[]; self.live_mode=True
        self.read_thread=threading.Thread(target=self.serial_reader,daemon=True)
        self.read_thread.start()
        self.master.after(60,self.update_live_plot)
        self.status_var.set("Live: ON")


    def serial_reader(self):
        while self.live_mode:
            try:
                line=self.serial_conn.readline().decode().strip()
                if line.replace('.','',1).isdigit():
                    self.live_buffer.append(float(line))
                if len(self.live_buffer)>5000:
                    self.live_buffer=self.live_buffer[-5000:]
            except: pass


    def update_live_plot(self):
        if not self.live_mode: return
        if len(self.live_buffer)>80:
            data=np.array(self.live_buffer)
            meanv,stdv,pc,bpm,sdnn,rmssd,peaks,filt=self.compute_metrics(data)
            t=np.linspace(0,len(filt)/self.fs,len(filt))
            self.ax.clear(); self.ax.plot(t,filt)
            if len(peaks)>0: self.ax.scatter(peaks/self.fs,filt[peaks],color='red')
            self.ax.set_title(f"Live HR={bpm:.1f}")

            label = self.auto_label(bpm, sdnn, rmssd)
            # Severity popup alert example (avoid too frequent popups on live, careful with UI)
            if "High" in label:
                self.status_var.set(f"Live HR={bpm:.1f} | {label} !!!")
            else:
                self.status_var.set(f"Live HR={bpm:.1f} | {label}")
                
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
        self.master.after(60,self.update_live_plot)


    def stop_live_read(self):
        self.live_mode=False
        try:
            if self.serial_conn: self.serial_conn.close()
        except: pass
        self.status_var.set("Live: OFF")


    def save_excel(self):
        path=filedialog.asksaveasfilename(defaultextension=".xlsx")
        if not path: return
        if len(self.live_buffer)>0:
            data=np.array(self.live_buffer)
            t=np.linspace(0,len(data)/self.fs,len(data))
            df=pd.DataFrame({"timestamp":t,"ecg_value":data})
        else:
            df=pd.DataFrame({"timestamp":self.time,"ecg_value":self.raw})
        df.to_excel(path,index=False)
        messagebox.showinfo("Saved","Excel saved")


    def generate_pdf(self):
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror("Missing","Install reportlab")
            return
        path=filedialog.asksaveasfilename(defaultextension=".pdf")
        if not path: return
        if len(self.live_buffer)>0: data=np.array(self.live_buffer)
        elif len(self.raw)>0: data=self.raw
        else: messagebox.showerror("No Data",""); return
        meanv,stdv,pc,bpm,sdnn,rmssd,peaks,filt=self.compute_metrics(data)
        label=self.auto_label(bpm,sdnn,rmssd)
        c=pdf_canvas.Canvas(path,pagesize=A4)
        w,h=A4
        c.setFont("Helvetica-Bold",16); c.drawString(50,h-50,"ECG Report")
        c.setFont("Helvetica",12)
        c.drawString(50,h-90,f"BPM: {bpm:.1f}")
        c.drawString(50,h-110,f"SDNN: {sdnn:.4f}")
        c.drawString(50,h-130,f"RMSSD: {rmssd:.4f}")
        c.drawString(50,h-150,f"Auto-label: {label}")
        c.save()
        messagebox.showinfo("Done","PDF generated")


    def train_ml(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Missing","Install scikit-learn")
            return
        path=filedialog.askopenfilename()
        if not path: return
        df=pd.read_csv(path)
        if 'ecg_value' not in df.columns:
            messagebox.showerror("Err","CSV must have ecg_value")
            return
        data=df['ecg_value'].values
        meanv,stdv,pc,bpm,sdnn,rmssd,peaks,filt=self.compute_metrics(data)
        label=self.auto_label(bpm,sdnn,rmssd)
        X=np.array([[meanv,stdv,pc,bpm,sdnn,rmssd]])
        y=np.array([label])
        model=RandomForestClassifier()
        model.fit(X,y)
        self.model=model
        pickle.dump(model,open("models/ecg_rf_autolabel.pkl","wb"))
        messagebox.showinfo("ML","Model trained with auto-label: "+label)


    def predict_ml(self):
        if self.model is None:
            if os.path.exists("models/ecg_rf_autolabel.pkl"):
                self.model=pickle.load(open("models/ecg_rf_autolabel.pkl","rb"))
            else:
                messagebox.showerror("No model","Train model first")
                return
        if len(self.live_buffer)>0: data=np.array(self.live_buffer)
        elif len(self.raw)>0: data=self.raw
        else:
            messagebox.showerror("No Data","")
            return
        meanv,stdv,pc,bpm,sdnn,rmssd,peaks,filt=self.compute_metrics(data)
        feat=np.array([[meanv,stdv,pc,bpm,sdnn,rmssd]])
        pred=self.model.predict(feat)[0]
        messagebox.showinfo("Prediction","ML Prediction: "+pred)


def main():
    root=tk.Tk()
    ECGApp(root)
    root.mainloop()


if __name__=='__main__':
    main()
