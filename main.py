import librosa
import soundfile as sf
import os


def convert_mp3_to_wav(input_mp3_path, output_wav_path, target_sample_rate=16000):
    try:
        audio, sr = librosa.load(input_mp3_path, sr=target_sample_rate)

        sf.write(output_wav_path, audio, target_sample_rate)
        print(f"Файл успешно конвертирован и сохранен в: {output_wav_path}")
    except Exception as e:
        print(f"Ошибка при конвертации файла: {e}")


def batch_convert_mp3_to_wav(input_folder, output_folder, target_sample_rate=16000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".mp3", ".wav"))
            convert_mp3_to_wav(input_path, output_path, target_sample_rate)


input_dir = "mp3_folder"
output_dir = "wav_folder"
batch_convert_mp3_to_wav(input_dir, output_dir)
