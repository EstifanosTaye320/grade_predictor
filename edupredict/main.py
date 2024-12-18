from data_preprocessing import load_and_preprocess_data
from check_assumptions import check_assumptions
from train_model import train_and_tune_model
from evaluate_model import evaluate_model

def main():
    while True:
        print("\n--- Linear Regression Project Menu ---")
        print("1. Preprocess Data")
        print("2. Check Assumptions")
        print("3. Train and Tune Model")
        print("4. Evaluate Model")
        print("5. Exit")

        choice = input("Enter the number corresponding to your choice: ")

        if choice == '1':
            print("\nRunning Data Preprocessing...")
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
            print("Data preprocessing complete.")

        elif choice == '2':
            print("\nRunning Assumption Checks...")
            check_assumptions()
            print("Assumption check complete. Please review the plots.")

        elif choice == '3':
            print("\nTraining and Tuning Model...")
            best_model = train_and_tune_model()
            print("Model training and tuning complete.")

        elif choice == '4':
            print("\nEvaluating Model...")
            evaluate_model()
            print("Model evaluation complete.")

        elif choice == '5':
            exit_choice = input("Are you sure you want to exit? (y/n): ")
            if exit_choice.lower() == 'y':
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Returning to menu...")

        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    main()
