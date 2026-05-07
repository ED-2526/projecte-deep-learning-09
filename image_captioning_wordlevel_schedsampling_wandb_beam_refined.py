import os
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import Inception_V3_Weights
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# Local cache for model/tokenizer downloads.
os.environ["HF_HOME"] = "./huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "./huggingface_cache"
os.environ["TORCH_HOME"] = "./torch_models"


@dataclass
class Config:
    dataset_path: str = r"C:\Users\xiaom\Downloads\DatasetX"
    images_subdir: str = "Images"
    captions_filename: str = "captions.txt"
    cache_file: str = "features_cache_training_refined.pkl"
    project_name: str = "image-captioning"
    run_name: str = "wordlevel-schedsampling-beam-refined"
    encoder_checkpoint: str = "best_encoder_schedsampling_beam_refined.pth"
    decoder_checkpoint: str = "best_decoder_schedsampling_beam_refined.pth"
    max_images: int = 8000
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 32
    embed_size: int = 512
    hidden_size: int = 512
    vocab_size: int = 5000
    min_frequency: int = 5
    max_caption_length: int = 40
    max_decode_length: int = 20
    beam_size: int = 3
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.3
    patience: int = 2
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.25
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def images_path(self) -> Path:
        return Path(self.dataset_path) / self.images_subdir

    @property
    def captions_path(self) -> Path:
        return Path(self.dataset_path) / self.captions_filename

    @property
    def features_cache_path(self) -> Path:
        return Path(self.dataset_path) / self.cache_file


def preprocess_caption(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"([.!,;?])", r" \1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"<start> {cleaned} <end>"


def load_captions(captions_path: Path) -> Dict[str, List[str]]:
    images_captions: Dict[str, List[str]] = {}
    with captions_path.open("r", encoding="utf-8") as file:
        next(file)
        for line in file:
            try:
                image_name, caption = line.split(",", 1)
            except ValueError:
                continue
            images_captions.setdefault(image_name.strip(), []).append(
                preprocess_caption(caption)
            )
    return images_captions


def build_wordlevel_tokenizer(
    captions: Sequence[str], vocab_size: int, min_frequency: int
) -> PreTrainedTokenizerFast:
    base_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    base_tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<start>", "<end>"],
    )
    base_tokenizer.train_from_iterator(captions, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<start>",
            "eos_token": "<end>",
        }
    )
    return tokenizer


def create_feature_extractor(device: str) -> nn.Module:
    cnn = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    cnn.fc = nn.Identity()
    return cnn.to(device).eval()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_or_load_features(
    image_names: Sequence[str], config: Config
) -> Dict[str, torch.Tensor]:
    cache_path = config.features_cache_path
    if cache_path.exists():
        with cache_path.open("rb") as file:
            return pickle.load(file)

    transform = build_transform()
    extractor = create_feature_extractor(config.device)
    features: Dict[str, torch.Tensor] = {}

    for image_name in tqdm(image_names, desc="Extracting CNN features"):
        image_path = config.images_path / image_name
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(config.device)
        with torch.no_grad():
            feature_vector = extractor(image_tensor)
        features[image_name] = feature_vector.cpu().numpy()

    with cache_path.open("wb") as file:
        pickle.dump(features, file)
    return features


class ImageCaptionDataset(Dataset):
    def __init__(self, image_features: Sequence, captions: Sequence[str]):
        self.image_features = image_features
        self.captions = captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        return torch.tensor(self.image_features[index]).float(), self.captions[index]


def create_collate_fn(tokenizer: PreTrainedTokenizerFast, max_length: int):
    def collate_fn(batch: Sequence[Tuple[torch.Tensor, str]]):
        features, captions = zip(*batch)
        tokenized = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return torch.stack(features, dim=0), tokenized["input_ids"]

    return collate_fn


class CNNEncoder(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = features.view(features.size(0), -1)
        return self.relu(self.fc(self.dropout(flattened)))


class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        pad_idx: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.img_to_hidden = nn.Linear(embed_size, hidden_size)

    def forward(self, captions: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embedding(captions))
        h0 = self.img_to_hidden(image_embeddings).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        return self.fc(outputs)

    def forward_scheduled(
        self,
        captions: torch.Tensor,
        image_embeddings: torch.Tensor,
        sampling_probability: float,
    ) -> torch.Tensor:
        batch_size, seq_len = captions.shape
        hidden = self.img_to_hidden(image_embeddings).unsqueeze(0).repeat(2, 1, 1)
        cell = torch.zeros_like(hidden)
        current_tokens = captions[:, 0].unsqueeze(1)
        logits_steps = []

        for step in range(seq_len - 1):
            embeddings = self.dropout(self.embedding(current_tokens))
            output, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
            step_logits = self.fc(output)
            logits_steps.append(step_logits)

            predicted_tokens = step_logits.argmax(dim=-1)
            teacher_tokens = captions[:, step + 1].unsqueeze(1)

            if sampling_probability <= 0:
                current_tokens = teacher_tokens
                continue

            use_model = torch.rand(batch_size, device=captions.device) < sampling_probability
            use_model = use_model.unsqueeze(1)
            current_tokens = torch.where(use_model, predicted_tokens, teacher_tokens)

        return torch.cat(logits_steps, dim=1)


def build_examples(
    image_names: Sequence[str],
    features_dict: Dict[str, torch.Tensor],
    images_captions: Dict[str, List[str]],
) -> Tuple[List, List[str]]:
    image_features = []
    captions = []
    for image_name in image_names:
        if image_name not in features_dict:
            continue
        for caption in images_captions[image_name]:
            image_features.append(features_dict[image_name])
            captions.append(caption)
    return image_features, captions


def scheduled_sampling_probability(config: Config, epoch_index: int) -> float:
    if config.epochs <= 1:
        return config.scheduled_sampling_end
    progress = epoch_index / max(config.epochs - 1, 1)
    delta = config.scheduled_sampling_end - config.scheduled_sampling_start
    return config.scheduled_sampling_start + delta * progress


def train_one_epoch(
    encoder: nn.Module,
    decoder: RNNDecoder,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
    sampling_probability: float,
) -> float:
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for image_features, captions in data_loader:
        image_features = image_features.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        image_embeddings = encoder(image_features)
        outputs = decoder.forward_scheduled(
            captions[:, :-1],
            image_embeddings,
            sampling_probability=sampling_probability,
        )
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def evaluate_loss(
    encoder: nn.Module,
    decoder: RNNDecoder,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
) -> float:
    encoder.eval()
    decoder.eval()
    running_loss = 0.0

    with torch.no_grad():
        for image_features, captions in data_loader:
            image_features = image_features.to(device)
            captions = captions.to(device)
            outputs = decoder(captions[:, :-1], encoder(image_features))
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            running_loss += loss.item()

    return running_loss / len(data_loader)


def generate_caption_beam_search(
    feature_vector,
    encoder: nn.Module,
    decoder: RNNDecoder,
    tokenizer: PreTrainedTokenizerFast,
    device: str,
    max_length: int,
    beam_size: int,
) -> List[str]:
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        image = torch.tensor(feature_vector).float().to(device)
        image_embedding = encoder(image)
        hidden = decoder.img_to_hidden(image_embedding).unsqueeze(0).repeat(2, 1, 1)
        cell = torch.zeros_like(hidden)

        start_token = tokenizer.bos_token_id
        end_token = tokenizer.eos_token_id
        beams = [(0.0, [start_token], hidden, cell)]
        completed_beams = []

        for _ in range(max_length):
            new_beams = []
            for score, sequence, beam_hidden, beam_cell in beams:
                if sequence[-1] == end_token:
                    completed_beams.append((score, sequence))
                    continue

                current_token = torch.tensor([[sequence[-1]]], device=device)
                embeddings = decoder.embedding(current_token)
                output, (next_hidden, next_cell) = decoder.lstm(
                    embeddings, (beam_hidden, beam_cell)
                )
                log_probs = torch.log_softmax(decoder.fc(output.squeeze(1)), dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_size)

                for branch_idx in range(beam_size):
                    token_id = top_indices[0][branch_idx].item()
                    token_log_prob = top_log_probs[0][branch_idx].item()
                    new_beams.append(
                        (
                            score + token_log_prob,
                            sequence + [token_id],
                            next_hidden.clone(),
                            next_cell.clone(),
                        )
                    )

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item[0], reverse=True)
            beams = new_beams[:beam_size]

        final_candidates = completed_beams + [(score, sequence) for score, sequence, _, _ in beams]
        final_candidates.sort(key=lambda item: item[0], reverse=True)
        best_sequence = final_candidates[0][1] if final_candidates else [start_token]

    generated_tokens = []
    for token_id in best_sequence[1:]:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token == "<end>":
            break
        generated_tokens.append(token)
    return generated_tokens


def compute_bleu_scores(
    sample_image_names: Sequence[str],
    features_dict: Dict[str, torch.Tensor],
    images_captions: Dict[str, List[str]],
    encoder: nn.Module,
    decoder: RNNDecoder,
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
) -> Tuple[float, float]:
    references = []
    hypotheses = []

    for image_name in sample_image_names:
        if image_name not in features_dict:
            continue
        generated = generate_caption_beam_search(
            features_dict[image_name],
            encoder,
            decoder,
            tokenizer,
            config.device,
            config.max_decode_length,
            config.beam_size,
        )
        references.append([caption.split() for caption in images_captions[image_name]])
        hypotheses.append(generated)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu4


def log_prediction_examples(
    image_names: Sequence[str],
    features_dict: Dict[str, torch.Tensor],
    images_captions: Dict[str, List[str]],
    encoder: nn.Module,
    decoder: RNNDecoder,
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
    num_examples: int = 5,
) -> None:
    rows = []
    for image_name in image_names[:num_examples]:
        if image_name not in features_dict:
            continue
        prediction = " ".join(
            generate_caption_beam_search(
                features_dict[image_name],
                encoder,
                decoder,
                tokenizer,
                config.device,
                config.max_decode_length,
                config.beam_size,
            )
        )
        ground_truth = " | ".join(images_captions[image_name])
        rows.append([image_name, prediction, ground_truth])

    if rows:
        wandb.log(
            {
                "prediction_samples": wandb.Table(
                    columns=["image", "prediction", "ground_truth"],
                    data=rows,
                )
            }
        )


def save_best_models(encoder: nn.Module, decoder: nn.Module, config: Config) -> None:
    torch.save(encoder.state_dict(), config.encoder_checkpoint)
    torch.save(decoder.state_dict(), config.decoder_checkpoint)


def load_best_models_if_available(
    encoder: nn.Module, decoder: nn.Module, config: Config
) -> bool:
    encoder_path = Path(config.encoder_checkpoint)
    decoder_path = Path(config.decoder_checkpoint)
    if not encoder_path.exists() or not decoder_path.exists():
        return False

    encoder.load_state_dict(torch.load(encoder_path, map_location=config.device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=config.device))
    return True


def main() -> None:
    config = Config()
    random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    wandb.init(project=config.project_name, name=config.run_name, config=asdict(config))

    print("Loading captions...")
    images_captions = load_captions(config.captions_path)

    image_names = list(images_captions.keys())[: config.max_images]
    all_captions = [caption for image_name in image_names for caption in images_captions[image_name]]

    print("Building WordLevel tokenizer...")
    tokenizer = build_wordlevel_tokenizer(
        all_captions,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
    )
    actual_vocab_size = len(tokenizer)
    pad_idx = tokenizer.pad_token_id

    print("Extracting or loading CNN features...")
    features_dict = extract_or_load_features(image_names, config)

    available_images = [image_name for image_name in image_names if image_name in features_dict]
    train_images, val_images = train_test_split(
        available_images,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    x_train, y_train = build_examples(train_images, features_dict, images_captions)
    x_val, y_val = build_examples(val_images, features_dict, images_captions)

    collate_fn = create_collate_fn(tokenizer, config.max_caption_length)
    train_loader = DataLoader(
        ImageCaptionDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ImageCaptionDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    encoder = CNNEncoder(config.embed_size, config.dropout).to(config.device)
    decoder = RNNDecoder(
        vocab_size=actual_vocab_size,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        pad_idx=pad_idx,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    wandb.config.update(
        {
            "tokenizer_type": "WordLevel",
            "scheduled_sampling_enabled": True,
            "actual_vocab_size": actual_vocab_size,
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_examples": len(x_train),
            "val_examples": len(x_val),
        }
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        sampling_probability = scheduled_sampling_probability(config, epoch)
        train_loss = train_one_epoch(
            encoder,
            decoder,
            train_loader,
            optimizer,
            criterion,
            config.device,
            actual_vocab_size,
            sampling_probability=sampling_probability,
        )
        val_loss = evaluate_loss(
            encoder,
            decoder,
            val_loader,
            criterion,
            config.device,
            actual_vocab_size,
        )
        bleu1, bleu4 = compute_bleu_scores(
            val_images[:100],
            features_dict,
            images_captions,
            encoder,
            decoder,
            tokenizer,
            config,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_best_models(encoder, decoder, config)
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"BLEU-1={bleu1:.4f} | BLEU-4={bleu4:.4f} | "
            f"ss_prob={sampling_probability:.3f} | "
            f"best_val_loss={best_val_loss:.4f} | "
            f"patience={patience_counter}/{config.patience}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu_1": bleu1,
                "bleu_4": bleu4,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "checkpoint_saved": int(improved),
                "beam_size": config.beam_size,
                "scheduled_sampling_probability": sampling_probability,
            }
        )

        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    restored_best = load_best_models_if_available(encoder, decoder, config)
    if restored_best:
        print("Loaded best checkpointed weights for final evaluation.")

    log_prediction_examples(
        val_images,
        features_dict,
        images_captions,
        encoder,
        decoder,
        tokenizer,
        config,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
