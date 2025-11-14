# test_tts.py
import pyttsx3, sys, platform, time

try:
    # prefer sane driver on each OS
    drv = None
    if platform.system() == "Windows":
        drv = "sapi5"
    elif platform.system() == "Darwin":
        drv = "nsss"
    else:
        drv = "espeak"  # linux

    if drv:
        engine = pyttsx3.init(driverName=drv)
    else:
        engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    print("Available voices:")
    for i, v in enumerate(voices):
        print(i, v.id, getattr(v, "name", ""))

    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    print("Speaking test phrase now...")
    engine.say("This is a test from pyttsx three. If you hear this, T T S works.")
    engine.runAndWait()
    print("Done.")
except Exception as e:
    print("TTS error:", e)
    sys.exit(1)
