import json
import sys

def get_optimum_or_bounds(json_path, target_instance):
    """
    Attempts to obtain the optimum (or bounds) of a JSSP instance from a JSON file.

    The JSON file should contain the following structure for each instance:
        name, jobs, machines, optimum, and bounds (with lower and upper bounds).

    If the optimum is available for the instance, it is printed. If the optimum is not found,
    the known bounds (if available) are printed. If neither the optimum nor bounds are found,
    an error message is printed, and the function exits.

    :param json_path: path to the JSON file
    :param target_instance: name of the instance to be searched
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error reading JSON: {e}")
            sys.exit(1)

        for instance in data:
            if instance.get('name') == target_instance:
                jobs = instance.get('jobs')
                machines = instance.get('machines')
                optimum = instance.get('optimum')

                if optimum is not None:
                    print(f"✅ Optimum for instance '{target_instance}': {optimum}")
                else:
                    bounds = instance.get('bounds', {})
                    lower = bounds.get('lower')
                    upper = bounds.get('upper')
                    if lower is not None or upper is not None:
                        print(f"ℹ️ Instance '{target_instance}' does not have a known optimum.")
                        print(f"ℹ️ Known bounds for '{target_instance}':")
                        print(f"   🔻 Lower Bound: {lower}")
                        print(f"   🔺 Upper Bound: {upper}")
                    else:
                        print(f"❌ No solution or bounds known for '{target_instance}'.")
                return

        print(f"❌ Instance '{target_instance}' not found in the JSON.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"\n❌ Incorrect usage.")
        print(f"👉 Correct format: python optimum.py <instance_name>\n")
        print(f"🔍 Example: python optimum.py abz5")
        sys.exit(1)

    target_instance = sys.argv[1].strip()

    if not target_instance:
        print("⚠️ Instance name cannot be empty.")
        sys.exit(1)

    get_optimum_or_bounds("responses.json", target_instance)
