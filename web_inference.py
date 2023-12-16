import os
import scipy.io.wavfile as wavf
import torch
from torch import no_grad, LongTensor
import tornado.ioloop
import tornado.web

import commons
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
import utils


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LANG_MARK = {
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

hps = None
net_g = None
speaker_id = None


class MainHandler(tornado.web.RequestHandler):

  def post(self):
    input_data = self.request.body.decode()

    if input_data.startswith('set'):
      config_path, model_path, spk = input_data[3:].split(',')
      self._set(config_path, model_path, spk)

    if input_data.startswith('generate'):
      language, output_name, length, text = input_data[8:].split(',')
      self._generate(language, output_name, float(length), text)

    self.write('')

  def _set(self, config_path: str, model_path: str, spk: str):
    global hps, net_g, speaker_id
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(DEVICE)
    net_g.eval()
    utils.load_checkpoint(model_path, net_g, None)
    speaker_id = hps.speakers[spk]

  def _generate(self, language: str, output_name: str, length: float, text:str):
    global hps, net_g, speaker_id
    text = f'{LANG_MARK[language]}{text}{LANG_MARK[language]}'
    stn_tst = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
      stn_tst = commons.intersperse(stn_tst, 0)
    stn_tst = LongTensor(stn_tst)

    with no_grad():
      x_tst = stn_tst.unsqueeze(0).to(DEVICE)
      x_tst_lengths = LongTensor([stn_tst.size(0)]).to(DEVICE)
      sid = LongTensor([speaker_id]).to(DEVICE)
      audio = net_g.infer(
          x_tst, x_tst_lengths,
          sid=sid,
          noise_scale=.667,
          noise_scale_w=0.6,
          length_scale=1.0 / length)[0][0, 0].data.cpu().float().numpy()

    wavf.write(
        os.path.join('.', 'output', 'vits', f'{output_name}.wav'),
        hps.data.sampling_rate,audio)


def Main():
  tornado.web.Application([
    (r"/", MainHandler),
  ]).listen(8080)
  tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
  Main()
