# Environment for `web_inference.py` on Windows

1. Run `python -m virtualenv venv` to create `.\venv`.

2. Run `.\venv\Scripts\activate` to activate the python virtual enviroment.

3. Run the following commands copied from the STEP 1 of [Google Colab](https://colab.research.google.com/drive/1pn1xnFfdLK63gVXDwV4zCXfVeo8c-I-0?usp=sharing):
    ```
    python -m pip install --upgrade --force-reinstall regex
    python -m pip install --force-reinstall soundfile
    python -m pip install --force-reinstall gradio
    python -m pip install imageio==2.4.1
    python -m pip install --upgrade youtube-dl
    python -m pip install moviepy
    python -m pip install --no-build-isolation -r requirements.txt
    python -m pip install --upgrade numpy
    python -m pip install --upgrade --force-reinstall numba
    python -m pip install --upgrade Cython
    python -m pip install --upgrade pyzmq
    python -m pip install pydantic==1.10.4
    python -m pip install ruamel.yaml
    python -m pip install git+https://github.com/openai/whisper.git

    cd monotonic_align/
    mkdir monotonic_align
    python setup.py build_ext --inplace
    ```

4. Install other python requirements:
    ```
    python -m pip install scipy
    python -m pip install librosa
    python -m pip install unidecode
    python -m pip install pyopenjtalk
    python -m pip install jamo
    python -m pip install ko_pron
    python -m pip install pypinyin
    python -m pip install jieba
    python -m pip install cn2an
    python -m pip install indic_transliteration
    python -m pip install inflect
    python -m pip install eng-to-ipa
    python -m pip install num_thai
    ```

5. Install Tornado by running: `python -m pip install tornado`

6. Run `..\VITS-fast-fine-tuning-edit\venv\Scripts\python.exe ..\VITS-fast-fine-tuning-edit\web_inference.py` in `Kyu-Discord-Bot` directory.
