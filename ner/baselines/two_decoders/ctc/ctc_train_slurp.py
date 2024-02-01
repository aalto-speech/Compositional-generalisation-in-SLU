import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from torchaudio.models.decoder import ctc_decoder
import sentencepiece as spm
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import string
import numpy as np
import tqdm
from jiwer import wer, cer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # resample to 16kHz
        wavs = self.hparams.resampler(wavs)
        
        wavlm_out = self.modules.wavlm(wavs.detach())
        # take the last layer
        wavlm_out = wavlm_out[-1]
        # Encoder pass
        encoder_out = self.modules.encoder(wavlm_out)

        # ASR
        ctc_logits = self.modules.ctc_lin(encoder_out)
        predictions = {"asr_preds": self.hparams.log_softmax(ctc_logits)}
        # NER
        ctc_logits_ner = self.modules.ner_lin(encoder_out)
        predictions["ner_preds"] = self.hparams.log_softmax(ctc_logits_ner)

        return predictions, lens


    def compute_objectives(self, predictions, batch, stage):
        # Compute losses
        predictions, lens = predictions
        tokens, token_lens = batch.encoded_chars
        ner_tokens, ner_token_lens = batch.encoded_ners

        asr_loss = self.hparams.ctc_cost(
            predictions["asr_preds"], 
            tokens, lens, 
            token_lens
        )
        ner_loss = self.hparams.ctc_cost(predictions["ner_preds"], 
            ner_tokens, 
            lens, 
            ner_token_lens
        )

        # loss = asr_loss + ner_loss
        alpha = 0.4
        loss = alpha * asr_loss + (1 - alpha) * ner_loss

        if stage != sb.Stage.TRAIN:
            ### ASR ###
            # Converted predicted tokens from indexes to words
            predicted_tokens = sb.decoders.ctc.ctc_greedy_decode(
                predictions["asr_preds"], 
                lens, 
                blank_id=self.hparams.blank_index
            )
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq))
                .split(" ") for utt_seq in predicted_tokens
            ]
            target_words = [words.split(" ") for words in batch.transcript]

            # Monitor word error rate and character error rated at valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)
            
            ### NER ###
            predicted_tokens_ner = sb.decoders.ctc.ctc_greedy_decode(
                predictions["ner_preds"], 
                lens, 
                blank_id=self.hparams.blank_index
            )
            predicted_ners = [
                "".join(self.ner_tokenizer.decode_ndim(ner_seq)).
                split(" ") for ner_seq in predicted_tokens_ner
            ]
            target_ners = [ners.split(" ") for ners in batch.ner]
            self.wer_metric_ner.append(batch.id, predicted_ners, target_ners)
            ### NER ###

        return loss
    

    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer() 
            self.cer_metric = self.hparams.cer_computer() 
            self.wer_metric_ner = self.hparams.error_rate_computer_ner() 


    def fit_batch(self, batch):
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        loss.backward()

        if self.check_gradients(loss):
            self.wavlm_optimizer.step()
            self.model_optimizer.step()
        self.wavlm_optimizer.zero_grad()
        self.model_optimizer.zero_grad()

        return loss.detach().cpu()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER_NER"] = self.wer_metric_ner.summarize("error_rate")


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wavlm, new_lr_wavlm = self.hparams.lr_annealing_wavlm(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wavlm_optimizer, new_lr_wavlm
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wavlm": old_lr_wavlm,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER_NER": stage_stats["WER_NER"]}, min_keys=["WER_NER"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            with open(self.hparams.decode_text_file, "w") as fo:
                for utt_details in self.wer_metric.scores:
                    print(utt_details["key"], " ".join
                        (utt_details["hyp_tokens"]), file=fo)
    

    def init_optimizers(self):
        "Initializes the wavlm optimizer and model optimizer"
        self.wavlm_optimizer = self.hparams.wavlm_opt_class(
            self.modules.wavlm.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wavlm_opt", self.wavlm_optimizer
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


    def zero_grad(self, set_to_none=False):
        self.wavlm_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)
    

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def is_ctc_active(self, stage):
        """Check if CTC is currently active.
        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current


        return current_epoch <= self.hparams.number_of_ctc_epochs
    

    def filter_predictions(self, predicted_words, batch):
        pred_labels = []
        true_labels = []
        for sent in predicted_words:
            pred_transcripts = []                    
            pred_ner = []
            if sent[-1] == "":
                sent = sent[:-1]
            for word in sent:
                word = word.rstrip()
                if word.startswith("<") and word.endswith(">"):
                    if "><" in word:
                        word = word.replace("><", "> <")
                        word = word.split()
                        ner_clean_1 = word[0]
                        ner_clean_2 = word[1]
                        pred_ner.append(ner_clean_1)
                        pred_ner.append(ner_clean_2)
                    else:
                        if "<" in word:
                            word = word.replace("<", " <")
                            word = word.replace(">", "> ")
                            words = word.split()
                            for w in words:
                                if w.startswith("<") and w.endswith(">"):
                                    pred_ner.append(w)
                                else:
                                    pred_transcripts.append(w)
                        else:
                            pred_ner.append(word)
                else:
                    if "<" in word:
                        word = word.replace("<", " <")
                        word = word.replace(">", "> ")
                        words = word.split()
                        for w in words:
                            if w.startswith("<") and w.endswith(">"):
                                pred_ner.append(w)
                            else:
                                pred_transcripts.append(w)
                    else:
                        pred_transcripts.append(word)
            sent = " ".join(pred_transcripts)
            pred_labels.append((sent, pred_ner))

        for sent in batch.transcript:
            true_transcripts = []
            true_ner = []
            for word in sent.split():
                if word.startswith("<") and word.endswith(">"):
                    true_ner.append(word)
                else:
                    true_transcripts.append(word)
            sent = " ".join(true_transcripts)
            true_labels.append((sent, true_ner))

        pred_trn = []
        pred_ner = []
        true_trn = []
        true_ner = []
        for sent, ner in pred_labels:
            pred_trn.append(sent)
            pred_ner.append(ner)
        for sent, ner in true_labels:
            true_trn.append(sent)
            true_ner.append(ner)

        return true_ner, pred_ner, true_trn, pred_trn


    def transcribe_dataset(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest WER
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        #self.on_evaluate_start(min_key=min_key) 
        # We call the on_evaluate_start that will load the best model
        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            true_labels_ner = []
            pred_labels_ner = []
            true_labels_trn = []
            pred_labels_trn = []
            #for batch in tqdm(dataset, dynamic_ncols=True):
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                out = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predictions, lens = out

                for sent in batch.transcript:
                    true_labels_trn.append(sent)

                # Greedy search
                predicted_tokens = sb.decoders.ctc.ctc_greedy_decode(
                    predictions["asr_preds"], 
                    lens, 
                    blank_id=self.hparams.blank_index
                )
                predicted_words = [
                    "".join(self.tokenizer.decode_ndim(utt_seq))
                    for utt_seq in predicted_tokens
                ]

                for sent in predicted_words:
                    pred_labels_trn.append(sent)
                # NER
                for ner_sent in batch.ner:
                    true_labels_ner.append(ner_sent)

                predicted_tokens_ner = sb.decoders.ctc.ctc_greedy_decode(
                    predictions["ner_preds"], 
                    lens, 
                    blank_id=self.hparams.blank_index
                )
                predicted_ner = [
                    "".join(self.ner_tokenizer.decode_ndim(utt_seq))
                    for utt_seq in predicted_tokens_ner
                ]
                for sent in predicted_ner:
                    pred_labels_ner.append(sent)

        mlb = MultiLabelBinarizer()
        true_ner_binary = mlb.fit_transform(true_labels_ner)
        pred_ner_binary = mlb.transform(pred_labels_ner)

        print("WER: ", wer(true_labels_trn, pred_labels_trn) * 100)
        print("CER: ", cer(true_labels_trn, pred_labels_trn) * 100)
        print("F1: ", f1_score(true_ner_binary, pred_ner_binary, average="micro") * 100)


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})

    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("data_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(data_path):
        sig = sb.dataio.dataio.read_audio(data_path)
        if len(sig.size()) == 2:
            sig = torch.mean(sig, dim=-1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    # label_encoder = sb.dataio.encoder.CTCTextEncoder()
    # label_encoder_ner = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript", "ner")
    @sb.utils.data_pipeline.provides("transcript", "ner", "encoded_ners", "encoded_chars")
    def text_pipeline(transcript, ner):
        yield transcript
        yield ner
        ner_list = ner.split()
        temp_ner_list = []
        for elem in ner_list:
            temp_ner_list.append(elem)
            temp_ner_list.append(" ")
        temp_ner_list = temp_ner_list[:-1]
        encoded_ners = torch.LongTensor(hparams["label_encoder_ner"].encode_sequence(temp_ner_list))
        yield encoded_ners
        char_list = []
        for word in transcript.split():
            word = word.lower()
            word = list(word)
            # replace punctuation and other non-vocab tokens with <unk>
            chars = []
            for char in word:
                if char in vocab:
                    chars.append(char)
                else:
                    # chars.append("<unk>")
                    pass
            char_list.extend(chars)
            char_list.extend(" ")

        encoded_chars = torch.LongTensor(hparams["label_encoder"].encode_sequence(char_list))
        yield encoded_chars

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # load the ner encoder
    #hparams["label_encoder_ner"].load_if_possible(hparams["label_encoder_ner_file"])

    # vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i",
    #          "j", "k", "l", "m", "n", "o", "p", "q", "r",
    #          "s", "t", "u", "v", "w", "x", "y", "z", " ", 
    #          "'"]

    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i",
             "j", "k", "l", "m", "n", "o", "p", "q", "r",
             "s", "t", "u", "v", "w", "x", "y", "z", " ", 
             "'", "-", "_", "<", ">"]

     # load the encoders
    hparams["label_encoder"].load_if_possible(hparams["label_encoder_file"])
    hparams["label_encoder_ner"].load_if_possible(hparams["label_encoder_ner_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, 
        [
            "id", 
            "sig", 
            "transcript", 
            "ner", 
            "encoded_ners", 
            "encoded_chars"
        ]
    )
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset creation
    train_data, valid_data, test_data = data_prep("../../data/ent_splits/easy_splits/two_decoders", hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

    # laod the pre-trained model if starting a new training
    if hparams["skip_training"] == False:
        print("Loading LibriSpeech model...")
        hparams["pretrainer"].collect_files(default_source=hparams["pre_trained_folder"])
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_brain.tokenizer = hparams["label_encoder"]
    asr_brain.ner_tokenizer = hparams["label_encoder_ner"]

    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###asr_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    else: 
        # evaluate
        print("Evaluating...")
        asr_brain.transcribe_dataset(test_data, "WER_NER", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
