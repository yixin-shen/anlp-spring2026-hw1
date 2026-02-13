import json

# 查看实际预测
with open('addition_models/best_model/predictions.json', 'r') as f:
    preds = json.load(f)

print("=== First 20 Predictions ===")
for i in range(min(20, len(preds))):
    pred_text = preds[i]
    try:
        parts = pred_text.split('=')
        question = parts[0]
        predicted_answer = parts[1] if len(parts) > 1 else "NO_ANSWER"

        # 计算正确答案
        nums = question.split('+')
        if len(nums) == 2:
            a, b = int(nums[0]), int(nums[1])
            correct_answer = a + b
            is_correct = (predicted_answer == str(correct_answer))
            print(f"{i}: {question}={predicted_answer} (correct: {correct_answer}) {'✓' if is_correct else '✗'}")
        else:
            print(f"{i}: {pred_text} (parse error)")
    except:
        print(f"{i}: {pred_text} (error)")
