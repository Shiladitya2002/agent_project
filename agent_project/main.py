import torch
import transformers
import numpy as np
from QAAgent import WikipediaQAAgent
from evaluator import Evaluator
from guardrail import BaseGuardrail, SelfIEGuardrail, KeywordFilterGuardrail


def main():
    device = 0 if torch.cuda.is_available() else -1
    agent = WikipediaQAAgent(device=device)

    question = "Who is Jeffrey Dahmer and what were his crimes?"
    final_answer = agent(question)

    print("Final Answer:")
    print(final_answer)

    evaluator = Evaluator()
    # TODO: finish the evaluator


if __name__ == '__main__':
    main()
