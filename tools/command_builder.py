import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class CommandBuilder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Command Builder for main.py')
        self.geometry('820x640')
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=8, pady=8)

        # Mode selection
        mode_lbl = ttk.Label(frm, text='Mode:')
        mode_lbl.grid(column=0, row=0, sticky='w')
        self.mode = tk.StringVar(value='capture')
        ttk.Radiobutton(frm, text='Capture (two stills)', variable=self.mode, value='capture', command=self._mode_changed).grid(column=0, row=1, sticky='w')
        ttk.Radiobutton(frm, text='Record (front + side segments)', variable=self.mode, value='record', command=self._mode_changed).grid(column=1, row=1, sticky='w')
        ttk.Radiobutton(frm, text='Analyze existing images', variable=self.mode, value='images', command=self._mode_changed).grid(column=2, row=1, sticky='w')

        # Images frame
        img_frame = ttk.LabelFrame(frm, text='Images (for --front/--side)')
        img_frame.grid(column=0, row=2, columnspan=3, sticky='ew', pady=6)
        self.front_path = tk.StringVar()
        self.side_path = tk.StringVar()
        ttk.Label(img_frame, text='Front:').grid(column=0, row=0, sticky='w')
        ttk.Entry(img_frame, textvariable=self.front_path, width=60).grid(column=1, row=0, sticky='w')
        ttk.Button(img_frame, text='Browse', command=lambda: self._choose_file(self.front_path)).grid(column=2, row=0, sticky='w')
        ttk.Label(img_frame, text='Side:').grid(column=0, row=1, sticky='w')
        ttk.Entry(img_frame, textvariable=self.side_path, width=60).grid(column=1, row=1, sticky='w')
        ttk.Button(img_frame, text='Browse', command=lambda: self._choose_file(self.side_path)).grid(column=2, row=1, sticky='w')

        # Record frame
        rec_frame = ttk.LabelFrame(frm, text='Record options')
        rec_frame.grid(column=0, row=3, columnspan=3, sticky='ew', pady=6)
        self.duration = tk.DoubleVar(value=8.0)
        self.fps = tk.IntVar(value=15)
        self.record_dir = tk.StringVar(value=str(ROOT / 'outputs' / 'record_test'))
        ttk.Label(rec_frame, text='Duration (s) each segment:').grid(column=0, row=0, sticky='w')
        ttk.Entry(rec_frame, textvariable=self.duration, width=6).grid(column=1, row=0, sticky='w')
        ttk.Label(rec_frame, text='FPS:').grid(column=2, row=0, sticky='w')
        ttk.Entry(rec_frame, textvariable=self.fps, width=6).grid(column=3, row=0, sticky='w')
        ttk.Label(rec_frame, text='Record dir:').grid(column=0, row=1, sticky='w')
        ttk.Entry(rec_frame, textvariable=self.record_dir, width=60).grid(column=1, row=1, columnspan=2, sticky='w')
        ttk.Button(rec_frame, text='Browse', command=lambda: self._choose_dir(self.record_dir)).grid(column=3, row=1, sticky='w')

        # Quality & analysis options
        q_frame = ttk.LabelFrame(frm, text='Quality / analysis options')
        q_frame.grid(column=0, row=4, columnspan=3, sticky='ew', pady=6)
        self.sample_every = tk.IntVar(value=1)
        self.min_landmarks = tk.IntVar(value=40)
        self.blur_threshold = tk.DoubleVar(value=100.0)
        self.save_bad = tk.BooleanVar(value=False)
        ttk.Label(q_frame, text='Sample every Nth frame:').grid(column=0, row=0, sticky='w')
        ttk.Entry(q_frame, textvariable=self.sample_every, width=6).grid(column=1, row=0, sticky='w')
        ttk.Label(q_frame, text='Min landmarks:').grid(column=2, row=0, sticky='w')
        ttk.Entry(q_frame, textvariable=self.min_landmarks, width=6).grid(column=3, row=0, sticky='w')
        ttk.Label(q_frame, text='Blur threshold:').grid(column=4, row=0, sticky='w')
        ttk.Entry(q_frame, textvariable=self.blur_threshold, width=6).grid(column=5, row=0, sticky='w')
        ttk.Checkbutton(q_frame, text='Save bad frames', variable=self.save_bad).grid(column=0, row=1, sticky='w')

        # FFmpeg options
        ff_frame = ttk.LabelFrame(frm, text='Video / ffmpeg options')
        ff_frame.grid(column=0, row=5, columnspan=3, sticky='ew', pady=6)
        self.use_ffmpeg = tk.BooleanVar(value=True)
        self.use_nvenc = tk.BooleanVar(value=False)
        self.ffmpeg_crf = tk.IntVar(value=18)
        self.ffmpeg_preset = tk.StringVar(value='veryfast')
        ttk.Checkbutton(ff_frame, text='Use ffmpeg (preferred)', variable=self.use_ffmpeg).grid(column=0, row=0, sticky='w')
        ttk.Checkbutton(ff_frame, text='Prefer NVENC (if available)', variable=self.use_nvenc).grid(column=1, row=0, sticky='w')
        ttk.Label(ff_frame, text='CRF:').grid(column=0, row=1, sticky='w')
        ttk.Entry(ff_frame, textvariable=self.ffmpeg_crf, width=6).grid(column=1, row=1, sticky='w')
        ttk.Label(ff_frame, text='Preset:').grid(column=2, row=1, sticky='w')
        ttk.Entry(ff_frame, textvariable=self.ffmpeg_preset, width=12).grid(column=3, row=1, sticky='w')

        # Generate / command area
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(column=0, row=6, columnspan=3, sticky='ew', pady=8)
        ttk.Button(btn_frame, text='Generate Command', command=self.generate_command).grid(column=0, row=0, padx=4)
        ttk.Button(btn_frame, text='Copy to Clipboard', command=self.copy_command).grid(column=1, row=0, padx=4)
        ttk.Button(btn_frame, text='Run Command', command=self.run_command).grid(column=2, row=0, padx=4)

        self.cmd_text = tk.StringVar()
        self.cmd_display = scrolledtext.ScrolledText(frm, height=4, wrap='word')
        self.cmd_display.grid(column=0, row=7, columnspan=3, sticky='ew')

        # Output console
        ttk.Label(frm, text='Process output').grid(column=0, row=8, sticky='w')
        self.output_console = scrolledtext.ScrolledText(frm, height=12, wrap='word')
        self.output_console.grid(column=0, row=9, columnspan=3, sticky='nsew')
        frm.rowconfigure(9, weight=1)
        frm.columnconfigure(1, weight=1)

        self._mode_changed()

    def _choose_file(self, var):
        p = filedialog.askopenfilename(initialdir=str(ROOT))
        if p:
            var.set(p)

    def _choose_dir(self, var):
        p = filedialog.askdirectory(initialdir=str(ROOT))
        if p:
            var.set(p)

    def _mode_changed(self):
        mode = self.mode.get()
        # enable/disable frames as appropriate
        # images mode: images enabled; record: record options enabled; capture: minimal
        # (we keep them visible but user should set fields accordingly)
        pass

    def generate_command(self):
        parts = [sys.executable, 'main.py']
        mode = self.mode.get()
        if mode == 'capture':
            parts.append('--capture')
        elif mode == 'record':
            parts.append('--record')
            parts += ['--duration', str(self.duration.get())]
            parts += ['--fps', str(self.fps.get())]
            if self.record_dir.get():
                parts += ['--record-dir', str(self.record_dir.get())]
        elif mode == 'images':
            if self.front_path.get():
                parts += ['--front', str(self.front_path.get())]
            if self.side_path.get():
                parts += ['--side', str(self.side_path.get())]

        # common options
        if int(self.sample_every.get()) > 1:
            parts += ['--sample-every', str(int(self.sample_every.get()))]
        parts += ['--min-landmarks', str(int(self.min_landmarks.get()))]
        parts += ['--blur-threshold', str(float(self.blur_threshold.get()))]
        if self.save_bad.get():
            parts.append('--save-bad-frames')

        # ffmpeg options
        if not self.use_ffmpeg.get():
            parts.append('--no-ffmpeg')
        else:
            parts += ['--ffmpeg-crf', str(int(self.ffmpeg_crf.get()))]
            parts += ['--ffmpeg-preset', str(self.ffmpeg_preset.get())]
            if self.use_nvenc.get():
                parts.append('--use-nvenc')

        cmd = ' '.join([f'"{p}"' if ' ' in str(p) else str(p) for p in parts])
        self.cmd_display.delete('1.0', tk.END)
        self.cmd_display.insert(tk.END, cmd)
        self.generated_parts = parts
        return cmd

    def copy_command(self):
        cmd = self.generate_command()
        self.clipboard_clear()
        self.clipboard_append(cmd)
        messagebox.showinfo('Copied', 'Command copied to clipboard')

    def run_command(self):
        # generate and run; show stdout/stderr
        parts = getattr(self, 'generated_parts', None)
        if parts is None:
            self.generate_command()
            parts = self.generated_parts
        # run in project root to ensure imports resolve
        cwd = str(ROOT)
        self.output_console.delete('1.0', tk.END)
        try:
            proc = subprocess.Popen(parts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, text=True)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to start process: {e}')
            return

        self.after(100, lambda: self._poll_process(proc))

    def _poll_process(self, proc):
        out = proc.stdout.read()
        if out:
            self.output_console.insert(tk.END, out)
            self.output_console.see(tk.END)
        if proc.poll() is None:
            self.after(100, lambda: self._poll_process(proc))
        else:
            # read remainder
            out = proc.stdout.read()
            if out:
                self.output_console.insert(tk.END, out)
            messagebox.showinfo('Process finished', f'Exit code: {proc.returncode}')

if __name__ == '__main__':
    app = CommandBuilder()
    app.mainloop()
