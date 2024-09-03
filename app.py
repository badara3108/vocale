
import io
import streamlit as st
import speech_recognition as sr
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationDetails
# Initialiser les clients API pour IBM et Microsoft
def initialize_ibm_client(api_key, url):
    authenticator = IAMAuthenticator(api_key)
    return SpeechToTextV1(authenticator=authenticator, url=url)

def initialize_microsoft_client(api_key, region):
    return SpeechConfig(subscription=api_key, region=region)

# Fonction pour transcrire avec Google
def transcribe_with_google(audio_data, recognizer, language):
    return recognizer.recognize_google(audio_data, language=language)

# Fonction pour transcrire avec IBM
def transcribe_with_ibm(audio_data, ibm_client):
    try:
        audio = io.BytesIO(audio_data.get_wav_data())
        response = ibm_client.recognize(
            audio=audio,
            content_type='audio/wav',
            model='en-US_BroadbandModel'
        ).get_result()
        return ' '.join(result['alternatives'][0]['transcript'] for result in response['results']), None
    except Exception as e:
        return None, f"Erreur IBM : {e}"

# Fonction pour transcrire avec Microsoft
def transcribe_with_microsoft(audio_data, microsoft_client):
    try:
        audio_stream = io.BytesIO(audio_data.get_wav_data())
        audio_config = AudioConfig(stream=audio_stream)
        recognizer = SpeechRecognizer(speech_config=microsoft_client, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()

        if result.reason == ResultReason.RecognizedSpeech:
            return result.text, None
        elif result.reason == ResultReason.NoMatch:
            return None, "Aucune correspondance trouvée."
        elif result.reason == ResultReason.Canceled:
            cancellation_details = CancellationDetails(result.cancellation_details)
            return None, f"Annulation : {cancellation_details.reason}"
    except Exception as e:
        return None, f"Erreur Microsoft : {e}"

# Fonction pour transcrire la parole en texte
def transcribe_speech(audio_file, recognizer, language, api_choice, ibm_client=None, microsoft_client=None):
    try:
        audio_data = sr.AudioFile(audio_file)
        with audio_data as source:
            audio = recognizer.record(source)

        if api_choice == "Google":
            return transcribe_with_google(audio, recognizer, language), None
        elif api_choice == "IBM":
            if ibm_client:
                return transcribe_with_ibm(audio, ibm_client)
            else:
                return None, "Client IBM non configuré correctement."
        elif api_choice == "Microsoft":
            if microsoft_client:
                return transcribe_with_microsoft(audio, microsoft_client)
            else:
                return None, "Client Microsoft non configuré correctement."
        else:
            raise ValueError("API non supportée")
    except sr.UnknownValueError:
        return None, "L'API de reconnaissance vocale n'a pas pu comprendre l'audio."
    except sr.RequestError as e:
        return None, f"Erreur de la demande à l'API de reconnaissance vocale : {e}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Erreur inconnue : {e}"

# Fonction pour enregistrer le texte dans un fichier
def save_text_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

# Fonction principale de l'application
def main():
    st.title("Application de Reconnaissance Vocale")

    # Déclaration des clients API
    ibm_client = None
    microsoft_client = None

    # Choisir l'API de reconnaissance vocale
    api_choice = st.selectbox("Choisissez l'API de reconnaissance vocale", ["Google", "IBM", "Microsoft"])

    # Configuration des clients API pour IBM et Microsoft
    if api_choice == "IBM":
        ibm_api_key = st.text_input("Clé API IBM", type="password")
        ibm_url = st.text_input("URL IBM")
        if ibm_api_key and ibm_url:
            try:
                ibm_client = initialize_ibm_client(ibm_api_key, ibm_url)
                st.success("Client IBM configuré avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de la configuration du client IBM : {e}")

    elif api_choice == "Microsoft":
        microsoft_api_key = st.text_input("Clé API Microsoft", type="password")
        microsoft_region = st.text_input("Région Microsoft")
        if microsoft_api_key and microsoft_region:
            try:
                microsoft_client = initialize_microsoft_client(microsoft_api_key, microsoft_region)
                st.success("Client Microsoft configuré avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de la configuration du client Microsoft : {e}")

    # Sélectionner la langue
    language = st.selectbox("Choisissez la langue", ["fr-FR", "en-US"])

    # Upload audio
    audio_file = st.file_uploader("Téléchargez un fichier audio", type=["wav", "mp3"])

    if audio_file:
        recognizer = sr.Recognizer()
        st.write("Fichier audio téléchargé. Prêt à transcrire.")

        if st.button("Transcrire"):
            st.write("Transcription en cours...")
            text, error = transcribe_speech(audio_file, recognizer, language, api_choice, ibm_client, microsoft_client)
            if error:
                st.error(error)
            else:
                st.write("Texte transcrit :")
                st.write(text)

                # Sauvegarder le texte dans un fichier
                if st.button("Enregistrer le texte dans un fichier"):
                    filename = st.text_input("Nom du fichier", "transcription.txt")
                    save_text_to_file(text, filename)
                    st.success(f"Texte enregistré dans {filename}")

if __name__ == "__main__":
    main()
