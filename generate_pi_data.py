import mpmath
from typing import List, Tuple

# Function to generate Pi up to a specific number of digits
def generate_pi_digits(num_digits: int) -> str:
    mpmath.mp.dps = num_digits  # Set precision
    pi = str(mpmath.mp.pi)[2:]  # Get the digits of Pi (skip "3.")
    return pi

# Function to create spans of digits for both training and validation sets
def generate_spans(pi_digits: str, span_size: int, validation_ratio: float) -> Tuple[List[str], List[str]]:
    spans = [pi_digits[i:i + span_size] for i in range(0, len(pi_digits), span_size)]
    
    # Determine the split point for training/validation sets based on the ratio
    split_point = int(len(spans) * (1 - validation_ratio))
    
    # Split the spans into training and validation sets
    training_set = spans[:split_point]
    validation_set = spans[split_point:]
    
    return training_set, validation_set

def main():
    num_digits = 1_000_001  # Total number of Pi digits to generate
    span_size = 100    # Span size for each training/validation piece
    validation_ratio = 0.2  # 20% of the spans will be used for validation

    # Generate Pi digits
    pi_digits = generate_pi_digits(num_digits)

    # Generate training and validation spans
    training_set, validation_set = generate_spans(pi_digits, span_size, validation_ratio)

    # Output the training and validation sets
    print("Training Set:")
    for i, span in enumerate(training_set):
        print(f"{i+1}: {span}")

    print("\nValidation Set:")
    for i, span in enumerate(validation_set):
        print(f"{i+1}: {span}")

    # save the training and validation sets to files
    with open("data/pi_train.txt", "w") as f:
        for span in training_set:
            f.write(span + "\n")

    with open("data/pi_val.txt", "w") as f:
        for span in validation_set:
            f.write(span + "\n")

if __name__ == "__main__":
    main()
