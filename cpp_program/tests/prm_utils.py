
import re

class ParameterFile:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.content = f.read()

    def set_value(self, subsection, key, value):
        # Regex to find the subsection
        # subsection Name ... end
        pattern = r"(subsection\s+" + re.escape(subsection) + r")(.*?)(end)"
        match = re.search(pattern, self.content, re.DOTALL)
        if not match:
            print(f"Warning: Subsection {subsection} not found")
            return

        subsection_content = match.group(2)
        
        # Regex to find 'set Key = Value' within the subsection
        # We need to be careful not to match other subsections if they were nested (usually not in deal.ii prm)
        
        key_pattern = r"(\s+set\s+" + re.escape(key) + r"\s*=\s*)([^#\n]+)"
        
        if re.search(key_pattern, subsection_content):
            new_sub_content = re.sub(key_pattern, f"\\g<1>{value}", subsection_content)
            self.content = self.content.replace(match.group(0), f"{match.group(1)}{new_sub_content}{match.group(3)}")
        else:
            print(f"Warning: Key {key} not found in {subsection}")

    def write(self, filepath):
        with open(filepath, 'w') as f:
            f.write(self.content)
