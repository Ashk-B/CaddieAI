import moviepy.editor as mp
import spacy
import speech_recognition as sr
from database_setup import InstructionalData, session

def extract_audio_from_video(video_path, audio_output_path="instructional_audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    print(f"Audio extracted to {audio_output_path}")
    return audio_output_path


def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print("Audio transcription complete")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


def extract_advice_from_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    advice_keywords = ["tip", "improve", "technique", "swing", "club"]
    extracted_advice = [sent.text for sent in doc.sents if any(word in sent.text.lower() for word in advice_keywords)]

    return extracted_advice


def store_instructional_advice(source, advice_text):
    try:
        advice_entry = InstructionalData(
            source=source,
            advice=advice_text
        )
        session.add(advice_entry)
        session.commit()
        print(f"Stored instructional advice from {source} into the database.")
    except Exception as e:
        session.rollback()
        print(f"Error storing instructional advice: {e}")


def process_instructional_video(video_path):
    try:
        # Step 1: Extract audio
        audio_path = extract_audio_from_video(video_path)

        # Step 2: Transcribe the audio to text
        transcribed_text = transcribe_audio(audio_path)

        if transcribed_text:
            # Step 3: Extract advice from the transcribed text
            extracted_advice = extract_advice_from_text(transcribed_text)

            # Combine the advice into a single string
            advice_text = "\n".join(extracted_advice)

            # Step 4: Store the advice in the database
            store_instructional_advice(video_path, advice_text)
        else:
            print(f"No text extracted from the video at {video_path}")
    except Exception as e:
        print(f"Error processing instructional video {video_path}: {e}")
