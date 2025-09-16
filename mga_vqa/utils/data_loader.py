"""
Data loading utilities for MGA-VQA.
Supports loading from various document VQA datasets.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torchvision.transforms as transforms
from transformers import AutoTokenizer


class DocumentVQADataset(Dataset):
    """
    Dataset class for Document VQA tasks.
    Supports DocVQA, FUNSD, CORD, and other document understanding datasets.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 image_size: Tuple[int, int] = (1024, 1024),
                 max_answer_length: int = 50,
                 vocab_file: Optional[str] = None):
        """
        Initialize DocumentVQA dataset.
        
        Args:
            data_dir: Directory containing dataset files
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (width, height)
            max_answer_length: Maximum length for answer sequences
            vocab_file: Optional vocabulary file path
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_answer_length = max_answer_length
        
        # Load dataset annotations
        self.annotations = self.load_annotations()
        
        # Load vocabulary
        self.vocab, self.vocab_inv = self.load_vocabulary(vocab_file)
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Tokenizer for answer encoding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def load_annotations(self) -> List[Dict]:
        """Load dataset annotations."""
        annotation_file = os.path.join(self.data_dir, f'{self.split}.json')
        
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        return annotations
    
    def load_vocabulary(self, vocab_file: Optional[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load or create vocabulary."""
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        else:
            # Create vocabulary from annotations
            vocab = self.build_vocabulary()
        
        # Create inverse vocabulary
        vocab_inv = {idx: token for token, idx in vocab.items()}
        
        return vocab, vocab_inv
    
    def build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from dataset annotations."""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for ann in self.annotations:
            # Tokenize answer
            answer_tokens = self.tokenizer.tokenize(ann.get('answer', '').lower())
            for token in answer_tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        return vocab
    
    def encode_answer(self, answer: str) -> List[int]:
        """Encode answer text to token ids."""
        if not answer:
            return [self.vocab['<PAD>']] * self.max_answer_length
        
        # Tokenize answer
        tokens = self.tokenizer.tokenize(answer.lower())
        
        # Convert to ids
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Add SOS and EOS tokens
        token_ids = [self.vocab['<SOS>']] + token_ids + [self.vocab['<EOS>']]
        
        # Pad or truncate to max length
        if len(token_ids) > self.max_answer_length:
            token_ids = token_ids[:self.max_answer_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (self.max_answer_length - len(token_ids)))
        
        return token_ids
    
    def normalize_bbox(self, bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """Normalize bounding box coordinates to [0, 1]."""
        x1, y1, x2, y2 = bbox
        return [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_path = os.path.join(self.data_dir, 'images', image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = Image.open(full_path).convert('RGB')
        return image
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        ann = self.annotations[idx]
        
        # Load image
        image = self.load_image(ann['image'])
        original_size = image.size  # (width, height)
        
        # Apply transforms
        image_tensor = self.image_transforms(image)
        
        # Get question
        question = ann['question']
        
        # Get OCR results
        ocr_results = ann.get('ocr_results', [])
        
        # Normalize OCR bounding boxes
        normalized_ocr = []
        for ocr_item in ocr_results:
            normalized_bbox = self.normalize_bbox(
                ocr_item['bbox'], original_size[0], original_size[1]
            )
            normalized_ocr.append({
                'text': ocr_item['text'],
                'bbox': normalized_bbox
            })
        
        # Encode answer
        answer = ann.get('answer', '')
        answer_tokens = self.encode_answer(answer)
        
        # Get answer bounding box (if available)
        answer_bbox = ann.get('answer_bbox')
        if answer_bbox:
            answer_bbox = self.normalize_bbox(
                answer_bbox, original_size[0], original_size[1]
            )
        else:
            answer_bbox = [0.0, 0.0, 1.0, 1.0]  # Default to full image
        
        return {
            'image': image_tensor,
            'question': question,
            'ocr_results': normalized_ocr,
            'image_size': original_size,
            'answer_labels': torch.tensor(answer_tokens, dtype=torch.long),
            'bbox_targets': torch.tensor(answer_bbox, dtype=torch.float32),
            'answer_text': answer,
            'image_id': ann.get('image_id', idx),
            'question_id': ann.get('question_id', idx)
        }


class DocVQACollator:
    """
    Collate function for DocumentVQA datasets.
    Handles batching of variable-length sequences and OCR results.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch items."""
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Collect questions
        questions = [item['question'] for item in batch]
        
        # Collect OCR results (list of lists)
        ocr_results = [item['ocr_results'] for item in batch]
        
        # Collect image sizes
        image_sizes = [item['image_size'] for item in batch]
        
        # Stack answer labels
        answer_labels = torch.stack([item['answer_labels'] for item in batch])
        
        # Stack bounding box targets
        bbox_targets = torch.stack([item['bbox_targets'] for item in batch])
        
        # Collect additional info
        answer_texts = [item['answer_text'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        question_ids = [item['question_id'] for item in batch]
        
        return {
            'image': images,
            'question': questions,
            'ocr_results': ocr_results,
            'image_sizes': image_sizes,
            'answer_labels': answer_labels,
            'bbox_targets': bbox_targets,
            'answer_texts': answer_texts,
            'image_ids': image_ids,
            'question_ids': question_ids
        }


def create_data_loaders(data_dir: str, config: Dict[str, Any],
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing dataset files
        config: Configuration dictionary
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DocumentVQADataset(
        data_dir=data_dir,
        split='train',
        image_size=(config.get('image_size', 1024), config.get('image_size', 1024)),
        max_answer_length=config.get('max_answer_length', 50)
    )
    
    val_dataset = DocumentVQADataset(
        data_dir=data_dir,
        split='val',
        image_size=(config.get('image_size', 1024), config.get('image_size', 1024)),
        max_answer_length=config.get('max_answer_length', 50)
    )
    
    test_dataset = DocumentVQADataset(
        data_dir=data_dir,
        split='test',
        image_size=(config.get('image_size', 1024), config.get('image_size', 1024)),
        max_answer_length=config.get('max_answer_length', 50)
    )
    
    # Create collator
    collator = DocVQACollator(pad_token_id=train_dataset.vocab['<PAD>'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class SyntheticVQADataset(Dataset):
    """
    Synthetic dataset for testing MGA-VQA components.
    Generates random document images with OCR results and questions.
    """
    
    def __init__(self, num_samples: int = 1000, image_size: Tuple[int, int] = (1024, 768)):
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Sample questions and answers
        self.questions = [
            "What is the company name?",
            "What is the total amount?",
            "What is the date?",
            "Who is the recipient?",
            "What is the invoice number?"
        ]
        
        self.answers = [
            "ABC Corporation",
            "$1,234.56",
            "2023-12-01",
            "John Doe",
            "INV-001"
        ]
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_ocr_results(self) -> List[Dict[str, Any]]:
        """Generate synthetic OCR results."""
        num_text_regions = np.random.randint(5, 20)
        ocr_results = []
        
        for _ in range(num_text_regions):
            # Random text
            text = np.random.choice([
                "Company Name", "Invoice", "Date", "Amount", "Total",
                "Address", "Phone", "Email", "Item", "Quantity"
            ])
            
            # Random bounding box (normalized)
            x1 = np.random.uniform(0, 0.8)
            y1 = np.random.uniform(0, 0.8)
            x2 = min(1.0, x1 + np.random.uniform(0.1, 0.2))
            y2 = min(1.0, y1 + np.random.uniform(0.05, 0.1))
            
            ocr_results.append({
                'text': text,
                'bbox': [x1, y1, x2, y2]
            })
        
        return ocr_results
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate synthetic sample."""
        # Create random image
        image = torch.randn(3, self.image_size[1], self.image_size[0]) * 0.1 + 0.5
        image = torch.clamp(image, 0, 1)
        
        # Normalize image
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(image)
        
        # Random question and answer
        q_idx = np.random.randint(len(self.questions))
        question = self.questions[q_idx]
        answer = self.answers[q_idx]
        
        # Generate OCR results
        ocr_results = self.generate_ocr_results()
        
        # Encode answer (simplified)
        answer_tokens = [1, 2, 3, 4, 0]  # Dummy tokens
        answer_tokens += [0] * (45)  # Pad to 50
        
        # Random answer bbox
        answer_bbox = [
            np.random.uniform(0, 0.5),
            np.random.uniform(0, 0.5),
            np.random.uniform(0.5, 1.0),
            np.random.uniform(0.5, 1.0)
        ]
        
        return {
            'image': image,
            'question': question,
            'ocr_results': ocr_results,
            'image_size': self.image_size,
            'answer_labels': torch.tensor(answer_tokens, dtype=torch.long),
            'bbox_targets': torch.tensor(answer_bbox, dtype=torch.float32),
            'answer_text': answer,
            'image_id': idx,
            'question_id': idx
        }


def create_synthetic_data_loader(batch_size: int = 16, num_samples: int = 100,
                               num_workers: int = 2) -> DataLoader:
    """Create synthetic data loader for testing."""
    dataset = SyntheticVQADataset(num_samples=num_samples)
    collator = DocVQACollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
