import model.evaluator as Evaluator

evaluator = Evaluator.eval_sentence()

evaluator.eval_sent(['O', 'O', 'O', 'S-gene', 'S-gene'], ['B-gene', 'I-gene', 'E-gene', 'S-gene', 'S-gene'])

print (evaluator.f1_score())