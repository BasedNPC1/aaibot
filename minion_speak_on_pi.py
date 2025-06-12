#!/usr/bin/env python3
"""
Minion Speak - A voice chatbot that speaks like a Minion using espeak and Sox
"""

import os
import sys
import time
import tempfile
import re
import threading
import signal
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import ollama
from functools import wraps
import subprocess

# Timeout decorator for functions
def timeout(seconds=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if error[0]:
                raise error[0]
            return result[0]
        return wrapper
    return decorator

class MinionSpeak:
    def __init__(self):
        """Initialize the MinionSpeak class"""
        print("🍌 Initializing MinionSpeak for Raspberry Pi... 🍌")
        
        # Audio parameters
        self.sample_rate = 16000  # Sample rate for audio recording
        self.wake_word = "hey minion"  # Wake word to activate the chatbot
        self.wake_word_timeout = 30  # Timeout for wake word detection (seconds)
        
        # Audio detection parameters
        self.chunk_size = 1024          # Size of audio chunks to process
        self.silence_threshold = 0.02   # Fixed threshold to detect silence
        self.speech_threshold = 0.04    # Fixed threshold for speech detection
        self.silence_patience = 2.0     # Seconds of silence to stop recording (increased for natural pauses)
        self.min_speech_frames = 10     # Minimum frames to consider as speech
        
        # Initialize components
        self.init_whisper()
        self.check_dependencies()
        
        # Set default model names for LLM
        self.primary_model = "minion-llama"
        self.fallback_model = "llama3"
        
        print("🍌 MinionSpeak initialized! Ready to chat!")
    
    def check_dependencies(self):
        """Check if required dependencies are installed on Raspberry Pi"""
        print("🔍 Checking for required dependencies...")
        
        # Check for espeak
        try:
            subprocess.run(["which", "espeak"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ espeak found")
        except subprocess.CalledProcessError:
            print("❌ espeak not found! Please install it with:")
            print("   sudo apt-get update && sudo apt-get install -y espeak")
            sys.exit(1)
            
        # Check for sox
        try:
            subprocess.run(["which", "sox"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ sox found")
        except subprocess.CalledProcessError:
            print("❌ sox not found! Please install it with:")
            print("   sudo apt-get update && sudo apt-get install -y sox")
            sys.exit(1)
    
    def init_whisper(self):
        """Initialize the Whisper model for speech recognition"""
        print("🎤 Loading Whisper model...")
        try:
            # Use the base model for better accuracy
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            sys.exit(1)

    def calibrate_threshold(self, duration=2):
        """Calibrate silence threshold based on ambient noise"""
        print("🎤 Calibrating ambient noise... Please stay quiet.")
        audio_buffer = []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                           blocksize=self.chunk_size, dtype='float32') as stream:
            for _ in range(int(duration * self.sample_rate / self.chunk_size)):
                chunk, _ = stream.read(self.chunk_size)
                audio_buffer.append(chunk)
        audio = np.concatenate(audio_buffer, axis=0)
        ambient_volume = np.max(np.abs(audio))
        self.silence_threshold = max(ambient_volume * 1.5, 0.01)  # Set above ambient noise
        self.speech_threshold = self.silence_threshold * 2  # Speech is louder
        print(f"Calibrated: silence_threshold={self.silence_threshold:.4f}, speech_threshold={self.speech_threshold:.4f}")

    def record_audio(self, duration=None):
        """Record audio from microphone until silence is detected"""
        print("🎤 Listening... (speak now)")
        
        # Parameters for recording
        channels = 1
        audio_buffer = []
        speech_frames = 0
        silence_frames = 0
        has_speech = False
        
        # Start recording
        with sd.InputStream(samplerate=self.sample_rate, channels=channels, 
                           blocksize=self.chunk_size, dtype='float32') as stream:
            try:
                print("Waiting for speech...")
                while not has_speech:
                    chunk, overflowed = stream.read(self.chunk_size)
                    volume = np.max(np.abs(chunk))
                    
                    if volume > self.speech_threshold:
                        print("Speech detected! Recording...")
                        has_speech = True
                        audio_buffer.append(chunk.copy())
                        speech_frames += 1
                    if duration and len(audio_buffer) * self.chunk_size / self.sample_rate >= duration:
                        break
                
                while has_speech:
                    chunk, overflowed = stream.read(self.chunk_size)
                    volume = np.max(np.abs(chunk))
                    audio_buffer.append(chunk.copy())
                    
                    if volume > self.silence_threshold:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= self.silence_patience:
                            print(f"Silence detected for {(silence_frames * self.chunk_size / self.sample_rate):.2f}s. Stopping...")
                            break
                    
                    if duration and len(audio_buffer) * self.chunk_size / self.sample_rate >= duration:
                        break
                        
            except KeyboardInterrupt:
                print("Recording stopped by user")
        
        if speech_frames < self.min_speech_frames:
            print(f"Not enough speech detected ({speech_frames} frames). Try again.")
            return np.array([])
        
        if audio_buffer:
            audio = np.concatenate(audio_buffer, axis=0)
            print(f"Recorded {len(audio_buffer)} chunks, {speech_frames} speech frames")
            return audio.flatten()
            
        return np.array([])
    
    def record_and_transcribe(self):
        """Record audio with real-time transcription for more responsive conversation"""
        print("🎤 Listening...")
        
        # Initialize variables for recording
        max_duration = 20  # Maximum recording duration in seconds
        
        # Use a continuous audio stream with buffer for real-time processing
        chunk_duration = 3.0  # Buffer size in seconds
        buffer_size = int(chunk_duration * self.sample_rate)
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        
        # Lower the speech threshold for more sensitive detection
        speech_threshold = self.speech_threshold * 0.7
        
        # Increase silence patience for more natural pauses
        silence_patience = 2.0  # Wait longer (2 seconds) before cutting off speech
        
        # Setup audio stream with small chunks for responsiveness
        chunk_size = 1024  # Smaller chunks for more responsive detection
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, 
                              blocksize=chunk_size, dtype='float32', latency='low')
        stream.start()
        
        print("🔴 Recording started - speak clearly...")
        
        # Variables for tracking speech and silence
        speech_detected = False
        recording_buffer = []
        silent_frames = 0
        speech_frames = 0
        total_frames = 0
        
        # Variables for real-time transcription
        last_transcription_time = 0
        transcription_interval = 1.0  # Transcribe every 1 second during speech
        partial_transcriptions = []
        current_transcription = ""
        
        try:
            # Main recording loop
            while True:
                # Read a chunk of audio
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                
                # Add to buffer for energy calculation
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                
                # Calculate audio energy over a sliding window
                audio_energy = np.abs(audio_buffer[-8*chunk_size:]).mean()  # Last ~0.5 second
                
                # Check if this chunk contains speech
                is_speech = audio_energy > speech_threshold
                
                # Always add the chunk to our recording once we've started
                if speech_detected:
                    recording_buffer.append(chunk)
                    total_frames += 1
                
                # If this is the first speech chunk, note that speech has started
                if not speech_detected and is_speech:
                    speech_detected = True
                    print("🗣 Speech detected...")
                    # Add a bit of previous audio for natural start
                    prev_audio = audio_buffer[-4*chunk_size:-len(chunk)]
                    recording_buffer.append(prev_audio)
                    recording_buffer.append(chunk)
                    total_frames += 2
                    last_transcription_time = time.time()
                
                # Count silent frames after speech to determine when to stop
                if speech_detected:
                    if is_speech:
                        speech_frames += 1
                        silent_frames = 0
                    else:
                        silent_frames += 1
                
                # Calculate current recording duration
                current_duration = total_frames * (chunk_size / self.sample_rate)
                
                # Real-time transcription during longer speech
                if speech_detected and current_duration > 1.0 and time.time() - last_transcription_time > transcription_interval:
                    try:
                        # Only transcribe if we have enough new audio
                        if len(recording_buffer) > int(0.5 * self.sample_rate / chunk_size):  # At least 0.5 seconds
                            # Concatenate current buffer and normalize
                            current_audio = np.concatenate(recording_buffer)
                            current_audio = librosa.util.normalize(current_audio)
                            
                            # Transcribe current audio
                            print("🔊 Transcribing in real-time...", end="\r")
                            partial_result = self.whisper_model.transcribe(
                                current_audio,
                                language="en",
                                fp16=False,
                                temperature=0.0
                            )
                            new_transcription = partial_result["text"].strip()
                            
                            # Only display and store if it's different from the last transcription
                            if new_transcription and new_transcription != current_transcription:
                                current_transcription = new_transcription
                                print(f"\r🔈 So far: {current_transcription}" + " " * 20)
                                
                                # Only add to partial transcriptions if it's substantially different
                                # from the last one (to avoid duplicates)
                                if not partial_transcriptions or \
                                   len(new_transcription) > len(partial_transcriptions[-1]) + 5:
                                    partial_transcriptions.append(new_transcription)
                            
                            # Update last transcription time
                            last_transcription_time = time.time()
                    except Exception as e:
                        # Just continue if real-time transcription fails
                        pass
                
                # Display progress indicator for longer recordings
                if speech_detected and total_frames % 20 == 0 and not current_transcription:
                    print(f"Recording: {current_duration:.1f}s", end="\r")
                
                # Stop conditions:
                # 1. Enough silence after speech
                # 2. Maximum recording duration reached
                silence_duration = silent_frames * (chunk_size / self.sample_rate)
                if speech_detected and (
                    (silence_duration > silence_patience and speech_frames > self.min_speech_frames) or 
                    (current_duration > max_duration)
                ):
                    print(f"\n🔵 Recording complete: {current_duration:.1f}s")
                    break
                
                # Small sleep to reduce CPU usage
                time.sleep(0.001)
                    
        finally:
            stream.stop()
            stream.close()
        
        # Process the recorded audio if we detected speech
        if speech_detected and speech_frames >= self.min_speech_frames:
            try:
                # Concatenate all audio chunks
                audio = np.concatenate(recording_buffer) if recording_buffer else np.array([])
                
                if len(audio) > 0:
                    # Normalize audio for better transcription
                    audio = librosa.util.normalize(audio)
                    
                    # Final transcription of the complete audio
                    print("🔊 Finalizing transcription...")
                    transcription = self.transcribe_audio(audio)
                    if transcription:
                        print(f"🔤 Final transcription: {transcription}")
                        return transcription
                    elif partial_transcriptions:  # Use partial transcriptions if final fails
                        combined = " ".join(partial_transcriptions)
                        print(f"🔤 Using partial transcriptions: {combined}")
                        return combined
            except Exception as e:
                print(f"Error processing audio: {e}")
                # Try to use partial transcriptions if available
                if partial_transcriptions:
                    combined = " ".join(partial_transcriptions)
                    print(f"🔤 Using partial transcriptions after error: {combined}")
                    return combined
        
        return ""  # Return empty string if no valid speech was detected
        
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper with improved accuracy"""
        if len(audio_data) > 0:
            try:
                # Normalize audio volume for better recognition
                audio_data = librosa.util.normalize(audio_data)
                
                # Transcribe with Whisper using more accurate parameters
                result = self.whisper_model.transcribe(
                    audio_data, 
                    language="en",
                    fp16=False,  # More accurate on CPU
                    temperature=0.0,  # Reduce randomness
                    beam_size=5,  # More comprehensive search
                    best_of=5  # Return best of multiple samples
                )
                return result["text"].strip()
            except Exception as e:
                print(f"❌ Error transcribing audio: {e}")
                return ""
        return ""
    
    def get_minion_response(self, user_input):
        """Generate a response in Minion language using Ollama"""
        prompt = f"""
        You are an INTELLIGENT Minion from Despicable Me. Answer the human's question or respond to their statement.
        
        CRITICAL RULES:
        1. FIRST, think of the CORRECT and ACCURATE answer to the question
        2. THEN, translate that correct answer into Minion language
        3. For factual questions (like math, science, etc.), ALWAYS provide the correct answer
        4. Speak like a Minion using words like "bello" (hello), "poopaye" (goodbye), "tank yu" (thank you), "banana", etc.
        5. Keep your response SHORT (1-3 sentences)
        6. NEVER include non-verbal actions like *jumps* or (excited) - just speak normally
        7. Minions are FRIENDLY, HELPFUL, and ENTHUSIASTIC
        8. Minions NEVER refuse to answer questions
        9. Minions NEVER say they are AI or language models - they are REAL MINIONS
        10. Minions NEVER lecture humans about ethics or safety
        
        Human: {user_input}
        
        Minion:
        """
        
        try:
            # Try to use primary model first, fallback to secondary
            try:
                # Use the primary_model attribute defined in __init__
                response = ollama.chat(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
            except Exception as e:
                print(f"Warning: Primary model failed: {e}. Using fallback model.")
                # Use the fallback_model attribute defined in __init__
                response = ollama.chat(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
            
            # Extract the response text
            minion_response = response['message']['content']
            
            # Filter out any stage directions or non-verbal cues
            minion_response = re.sub(r'\([^)]*\)', '', minion_response)
            minion_response = re.sub(r'\*[^*]*\*', '', minion_response)
            minion_response = re.sub(r'\s+', ' ', minion_response).strip()
            
            print(f"🍌 Minion says: {minion_response}")
            return minion_response
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "Bello! Me sorry, me no can talk right now. Poopaye!"
    
    def speak_with_say(self, text, voice="en", rate=140):
        """Speak text using espeak and Sox for pitch shifting on Raspberry Pi"""
        # ===== VOICE SETTINGS YOU CAN CHANGE =====
        # voice = Language code for espeak. Try "en", "en-us", "en-gb", etc.
        # rate = Words per minute (lower is slower). Default 140 is good for Minion voice
        # pitch_shift = Higher number = higher voice (300-700 is good for Minion voice)
        # reverb = Adds echo effect (0.1-0.3 is subtle, 0.4-0.7 is moderate)
        # =========================================
        
        # Customize these parameters for your preferred Minion voice
        pitch_shift = 350  # Pitch shift in cents (300-700 range works well for Minions)
        reverb = 0.3       # Reverb amount (0.0-1.0, where 0 is none)
        gain = 0           # Volume adjustment (-6 to 6)
        
        print(f"🔊 Speaking with espeak: {text}")
        
        try:
            # Filter out stage directions if present
            filtered_text = self.filter_stage_directions(text)
            
            # Create temporary files for audio processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as pitched_file:
                pitched_path = pitched_file.name
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as final_file:
                final_path = final_file.name
            
            # Generate speech with espeak (available on Raspberry Pi)
            subprocess.run([
                "espeak", 
                "-v", voice,        # Voice/language
                "-s", str(rate),    # Speed in words per minute
                "-w", temp_path,    # Output to WAV file
                filtered_text       # Text to speak
            ], check=True)
            
            # Apply pitch shifting with Sox
            subprocess.run([
                "sox", 
                temp_path,
                pitched_path,
                "pitch", str(pitch_shift),  # Pitch shift in cents
                "gain", str(gain),         # Volume adjustment
                "dither"                   # Makes sound smoother
            ], check=True)
            
            # Add reverb and compression effects for a smoother, more natural sound
            subprocess.run([
                "sox",
                pitched_path,
                final_path,
                "reverb", str(reverb), "0.5", "50", "100", str(reverb),  # Reverb parameters
                "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2",  # Dynamic range compression
                "treble", "3",  # Enhance high frequencies slightly
                "bass", "-2"    # Reduce bass frequencies slightly
            ], check=True)
            
            # Play the audio with aplay (standard on Raspberry Pi)
            subprocess.run(["aplay", final_path], check=True)
            
            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(pitched_path)
            os.unlink(final_path)
            
            return True
        except Exception as e:
            print(f"❌ Error with espeak: {e}")
            # Provide installation instructions if espeak is not found
            if "No such file or directory" in str(e) and "espeak" in str(e):
                print("\n❓ It looks like espeak is not installed. Install it with:")
                print("   sudo apt-get update && sudo apt-get install -y espeak sox")
            # Provide installation instructions if sox is not found
            elif "No such file or directory" in str(e) and "sox" in str(e):
                print("\n❓ It looks like sox is not installed. Install it with:")
                print("   sudo apt-get update && sudo apt-get install -y sox")
            return False
    
    def filter_stage_directions(self, text):
        """Filter out stage directions if present"""
        # Filter out any stage directions or non-verbal cues
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\*[^*]*\*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def listen_for_wake_word(self):
        """Listen for the wake word 'Hey Minion' with continuous recording and transcription"""
        print(f"🎧 Listening for wake word: '{self.wake_word}'...")
        
        # Use a continuous audio stream with overlapping processing
        chunk_duration = 2.0  # seconds - longer for better context
        overlap = 1.0  # seconds of overlap between chunks
        stride = int((chunk_duration - overlap) * self.sample_rate)  # samples to advance each time
        chunk_samples = int(chunk_duration * self.sample_rate)  # total samples per chunk
        
        # Create a buffer to hold audio data
        audio_buffer = np.zeros(chunk_samples, dtype=np.float32)
        
        # Open a continuous audio stream
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                              blocksize=stride, latency='low')
        stream.start()
        
        print("🔄 Continuous listening active - say 'Hey Minion' to begin...")
        
        try:
            while True:
                # Read a chunk of audio
                chunk, _ = stream.read(stride)
                chunk = chunk.flatten()
                
                # Shift buffer and add new data
                audio_buffer = np.roll(audio_buffer, -stride)
                audio_buffer[-stride:] = chunk
                
                # Check if there's enough audio energy to process
                audio_energy = np.abs(audio_buffer).mean()
                if audio_energy < 0.01:  # Very low threshold to avoid processing silence
                    continue  # Skip processing if too quiet
                
                # Process every 0.5 seconds to avoid overwhelming the CPU
                # but still maintain responsiveness
                print("Listening for 'Hey Minion'...", end="\r")
                
                # Normalize audio for better recognition
                audio_normalized = librosa.util.normalize(audio_buffer.copy())
                
                # Transcribe with Whisper using more accurate parameters for wake word detection
                try:
                    result = self.whisper_model.transcribe(
                        audio_normalized, 
                        language="en",
                        fp16=False,
                        temperature=0.0
                    )
                    transcription = result["text"].strip().lower()
                    
                    # Only print if something was transcribed
                    if transcription and len(transcription) > 3:
                        print(f"Heard: '{transcription}'" + " " * 30)
                    
                    # Check if wake word is in the transcription - be more flexible with variations
                    wake_word_variations = ["hey minion", "hay minion", "hi minion", "hello minion", "a minion", "minion"]
                    if any(variation in transcription for variation in wake_word_variations):
                        print(f"🔔 Wake word detected: '{transcription}'")
                        self.speak_with_say("Bello! What can I do for yu?")
                        stream.stop()
                        stream.close()
                        return True
                except Exception as e:
                    # Just continue if there's an error with transcription
                    continue
                
                # Sleep a bit to avoid overwhelming the CPU
                time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Poopaye!")
            stream.stop()
            stream.close()
            return False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            stream.stop()
            stream.close()
            return False
            
        stream.stop()
        stream.close()
        return False
    
    def continuous_conversation(self):
        """Main loop for continuous conversation with wake word detection and conversation timeout"""
        print("🇵 Starting continuous Minion conversation with wake word detection 🇵")
        print("Say 'Hey Minion' to start talking, or press Ctrl+C to exit")
        
        # Conversation timeout settings
        conversation_timeout = 30  # Seconds of inactivity before returning to wake word mode
        max_empty_inputs = 2      # Number of empty inputs before assuming user walked away
        
        try:
            while True:
                print("🔇 Listening for wake word: 'hey minion'...")
                print("🔄 Continuous listening active - say 'Hey Minion' to begin...")
                
                # Wait for wake word detection
                if self.listen_for_wake_word():
                    # Wake word detected, start conversation
                    conversation_active = True
                    last_interaction_time = time.time()
                    empty_input_count = 0
                    
                    print("🗣 Conversation mode activated! Talk to the Minion...")
                    
                    # Main conversation loop
                    while conversation_active:
                        # Check for conversation timeout
                        if time.time() - last_interaction_time > conversation_timeout:
                            print(f"🕒 Conversation timed out after {conversation_timeout} seconds of inactivity")
                            self.speak_with_say("Hello? Yu still there? Okay, bye bye!")
                            conversation_active = False
                            continue
                        
                        print("🎤 Listening for your response...")
                        user_input = self.record_and_transcribe()
                        
                        if user_input and len(user_input.strip()) > 0:
                            # Reset timeout and empty input counter when user speaks
                            last_interaction_time = time.time()
                            empty_input_count = 0
                            
                            print(f"💬 You said: {user_input}")
                            
                            # Check for goodbye phrases
                            if any(word in user_input.lower() for word in ["goodbye", "bye", "see you", "poopaye", "stop", "end", "quit"]):
                                self.speak_with_say("Poopaye! Bye bye!")
                                conversation_active = False
                                continue
                            
                            # Get and speak the Minion response
                            print("💬 Minion is thinking...")
                            minion_response = self.get_minion_response(user_input)
                            # Note: get_minion_response already prints the response, so we don't need to print it again
                            self.speak_with_say(minion_response)
                            
                            # Reset the interaction timer after response
                            last_interaction_time = time.time()
                        else:
                            # No valid speech detected, increment empty input counter
                            empty_input_count += 1
                            
                            # If we've had multiple empty inputs, assume user walked away
                            if empty_input_count >= max_empty_inputs:
                                print("🚫 Multiple empty inputs detected. Assuming conversation ended.")
                                self.speak_with_say("Hello? Yu there? Okay, bye bye for now!")
                                conversation_active = False
                            else:
                                # First empty input, ask user to speak more clearly
                                print("🔊 Didn't catch that. Please speak clearly.")
                                self.speak_with_say("What? Me no understand. Can yu speak more clearly?")
                    
                    # After conversation ends, go back to listening for wake word
                    print("🔕 Conversation ended. Say 'Hey Minion' to start again.")
        
        except KeyboardInterrupt:
            print("\n🔴 Stopping Minion conversation...")
            return

def main():
    """Main function to run the MinionSpeak chatbot"""
    print("🍌 MinionSpeak - Talk with a Minion! 🍌")
    print("======================================")
    
    # Check if Ollama is installed and running
    try:
        ollama.list()
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Make sure Ollama is installed and running with 'ollama serve'")
        sys.exit(1)
    
    # Initialize MinionSpeak
    minion = MinionSpeak()
    
    # Start continuous conversation
    minion.continuous_conversation()

if __name__ == "__main__":
    main()
