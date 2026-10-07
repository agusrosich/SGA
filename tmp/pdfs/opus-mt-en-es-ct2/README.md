---
license: apache-2.0
tags:
- translation
- opus-mt
- ctranslate2
- quantized
language:
- multilingual
pipeline_tag: translation
---

# opus-mt-en-es-ctranslate2-android

This is a quantized INT8 version of `Helsinki-NLP/opus-mt-en-es` converted to CTranslate2 format for efficient inference.

## Model Details

- **Original Model**: Helsinki-NLP/opus-mt-en-es
- **Format**: CTranslate2
- **Quantization**: INT8
- **Framework**: OPUS-MT
- **Converted by**: Automated conversion pipeline

## Usage

### With CTranslate2

```python
import ctranslate2
import sentencepiece as spm

# Load the model
translator = ctranslate2.Translator("path/to/model")

# Load tokenizers
sp_source = spm.SentencePieceProcessor(model_file="source.spm")
sp_target = spm.SentencePieceProcessor(model_file="target.spm")

# Translate
source_tokens = sp_source.encode("Your text here", out_type=str)
results = translator.translate_batch([source_tokens])
translation = sp_target.decode(results[0].hypotheses[0])
```

## Performance

This INT8 quantized version provides:
- ~75% reduction in model size
- Faster inference speed
- Maintained translation quality
- Mobile-friendly deployment

## Original Model

Based on the OPUS-MT project: https://github.com/Helsinki-NLP/Opus-MT
