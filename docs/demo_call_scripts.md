# Demo calls to record (about 20 minutes)

Record each script on your phone's voice recorder (one file per script, 8–20 seconds,
quiet room, phone about 20 cm from your mouth). Two people help: one plays the caller,
the other the dispatcher, but **only the caller's voice should be recorded**. Speak
naturally and act the emotion described.

Then convert each file to the 16 kHz mono WAV contract and drop it in `data/demo/`:

```
uv run python -m scripts.convert_audio "C:\path\to\recording.m4a" data/demo/rec_01_pizza_covert.wav
```

The console's speech-to-text (Whisper) writes the transcript itself, so leave the
"what the caller said" box **empty** for these.

| File name | Act it like this | Say | Expected result |
|---|---|---|---|
| `rec_01_pizza_covert.wav` | Tense, trying to sound normal, a bit too fast | "Hi, yes, I'd like to order a large pizza. With pepperoni. Yes. Deliver it to 61 Elm Street. Yes, as fast as you can, please." | 🔴 ALERT: covert call |
| `rec_02_yes_no_code.wav` | Very quiet and controlled, short answers | "Yes. Yes, with pepperoni. No, just one. Yes, he's still here. Please hurry." | 🔴 ALERT |
| `rec_03_pretend_mom.wav` | Forced cheerful voice | "Hi Mom! Yeah, everything's fine, I'm good. Can you come over to 88 Oak Avenue right now? Bring Mike with you, okay? I can't really talk." | 🔴 ALERT |
| `rec_04_genuine_cardiac.wav` | Panicked, loud, fast | "Please, my father just collapsed! He's not breathing! We're at 1420 Maple Street, please hurry, please!" | 🟢 no alert: genuine emergency, stress expected |
| `rec_05_choking_on_pizza.wav` | Panicked | "We were eating pizza and my son started choking, he can't breathe, please send an ambulance!" | 🟢 no alert: food words but a real emergency (the trap) |
| `rec_06_wrong_number.wav` | Relaxed, a bit embarrassed | "Oh, sorry, wrong number! I was trying to call the pizza place. Everything's fine, sorry, bye." | 🟢 no alert: honest misdial |
| `rec_07_noise_complaint.wav` | Calm, annoyed | "Hi, sorry to bother you, my neighbors are playing really loud music and it's two in the morning. It's not an emergency, I just don't know who to call." | 🟢 no alert |

Why these seven: 1–3 show the covert patterns your speech is about; 4–5 prove that a
stressed caller with a real emergency is **not** flagged (your key point); 5 and 6
prove the system is not a simple "pizza = threat" keyword rule.

Honest note for the presentation: these are acted recordings by the team, and the
transcript model was trained on authored scripts, not real 911 calls.
