---
license: mit
task_categories:
- question-answering
- text-generation
language:
- kk
size_categories:
- 10K<n<100K
---
# Dataset Card for Dataset Name

Self-instruct data pairs for Kazakh language


## Dataset Details

The dataset is translated from Standford Alpaca instruction dataset via Google Translations API.

1. Manually fixed the translation error.
2. Common names and places of Kazakhstan were added.
3. Intructions of kazakhstan history and cultures were added.

### Dataset Description




- **Curated by:** Mussa Aman
- **Language(s) (NLP):** Kazakh
- **License:** MIT


## Uses

This dataset is curated to fine-tune the LLaMA 2 model for the Kazakh language. It aims to enhance the model's understanding and processing capabilities of Kazakh, addressing a gap in the Low Resource Lanuguages for solving the NLP resources for Kazakh language. 

The dataset includes the self-instruct approach, there is commonly one "instruction","input" and "output" which is crucial for improving language comprehension and task performance of the model.


## Citation

**BibTeX:**

@inproceedings{mussa2025parameter,
  title={Parameter-Efficient Fine-Tuning of LLaMA 2 for the Kazakh Language: Advancing Low-Resource Language Models},
  author={Mussa, Aman and Mansurova, Madina},
  booktitle={International Congress on Information and Communication Technology},
  pages={511--520},
  year={2025},
  organization={Springer}
}
**APA:**

Mussa, A., & Mansurova, M. (2025, February). Parameter-Efficient Fine-Tuning of LLaMA 2 for the Kazakh Language: Advancing Low-Resource Language Models. In International Congress on Information and Communication Technology (pp. 511-520). Singapore: Springer Nature Singapore.



## Dataset Card Contact

Please contact in email: mussa.aman@kaznu.kz