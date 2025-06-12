# Minion Chatbot for Raspberry Pi

This is a voice-activated Minion chatbot that runs on Raspberry Pi. It listens for the wake word "Hey Minion", transcribes your speech, generates Minion-style responses, and speaks back with a Minion voice.

## Features

- Wake word detection ("Hey Minion")
- Real-time speech transcription using OpenAI Whisper
- Minion-style responses using Ollama LLM
- Minion voice synthesis using espeak and Sox for pitch shifting
- Continuous conversation with natural pauses
- Automatic conversation timeout after inactivity

## Requirements

- Raspberry Pi (3 or newer recommended)
- Microphone
- Speaker
- Internet connection (for LLM)

## Installation

1. Install required system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv espeak sox
```

2. Clone this repository:

```bash
git clone https://github.com/yourusername/bearbrickai.git
cd bearbrickai
```

3. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Install Ollama on your Raspberry Pi:
   Follow instructions at https://ollama.com/download

5. Pull the required models:

```bash
ollama pull minion-llama  # Primary model
ollama pull llama3        # Fallback model
```

## Running the Chatbot

1. Make sure Ollama is running:

```bash
ollama serve
```

2. In a new terminal, activate the virtual environment and run the chatbot:

```bash
source venv/bin/activate
python minion_speak_on_pi.py
```

3. Say "Hey Minion" to activate the chatbot, then speak your question or request.

## Voice Customization

You can customize the Minion voice by modifying these parameters in the `speak_with_say` method:

- `voice`: Language code for espeak (default: "en")
- `rate`: Words per minute (default: 140)
- `pitch_shift`: Pitch shift in cents (default: 350, range: 300-700)
- `reverb`: Reverb amount (default: 0.3, range: 0.0-1.0)
- `gain`: Volume adjustment (default: 0, range: -6 to 6)

## Troubleshooting

- **No sound**: Check your speaker connections and volume settings
- **Microphone not working**: Run `arecord -l` to list available audio devices
- **Ollama errors**: Make sure Ollama is running with `ollama serve`
- **High CPU usage**: Lower the Whisper model size in the code (change "base" to "tiny")

## Credits

- OpenAI Whisper for speech recognition
- Ollama for local LLM inference
- espeak and Sox for voice synthesis
