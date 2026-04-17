# Third-Party Notices

This project depends on third-party open-source software. The following notices are provided for attribution and license compliance.

## OpenAI CLIP
- Project: CLIP (Contrastive Language–Image Pre-training)
- License: MIT License
- Copyright: Copyright (c) 2021 OpenAI
- Used for: text/image embedding model and tokenization utilities

## PyTorch
- Project: PyTorch
- License: BSD-style (see PyTorch repository for full text)
- Used for: model execution and tensor operations

## NumPy
- Project: NumPy
- License: BSD-style
- Used for: numeric operations

## Pillow
- Project: Pillow
- License: HPND (PIL-style)
- Used for: image loading and preprocessing

## FastAPI
- Project: FastAPI
- License: MIT License
- Used for: HTTP API framework for the /score and /suggest endpoints

## Uvicorn
- Project: Uvicorn
- License: BSD 3-Clause License
- Used for: ASGI server to run the FastAPI application

## python-multipart
- Project: python-multipart
- License: Apache License 2.0
- Used for: parsing multipart/form-data uploads in the /score endpoint

## Pydantic
- Project: Pydantic
- License: MIT License
- Used for: request/response validation and schema definitions

## OpenAI Python Library
- Project: openai-python
- License: Apache License 2.0
- Used for: client for the OpenAI API (LLM feedback generation in /suggest)