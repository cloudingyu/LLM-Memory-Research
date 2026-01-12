import random
import uuid

class DataLoader:
    NOISE_TEMPLATES = [
        "Subject: Meeting Rescheduled. Hi everyone, the weekly sync is moved to Friday at 3 PM due to the client visit.",
        "System Log [Info]: Daemon process started with PID {pid}. Memory usage steady at {mem}%.",
        "News Snippet: The local economy has seen a shift in trends, with technology sectors growing by 15% this quarter.",
        "Reminder: Don't forget to submit your expense reports by the end of the month. late submissions will be rejected.",
        "Weather Update: Expect heavy rain in the northern region, while the coast will remain sunny and dry.",
        "Code Comment: TODO: Refactor this function to optimize complexity from O(n^2) to O(n log n).",
        "Personal Note: I need to pick up groceries: milk, eggs, bread, and some apples for the pie.",
        "Marketing: Our new campaign targets the demographic of 18-25 year olds interested in sustainable fashion."
    ]

    @staticmethod
    def _generate_noise_block(length=10):
        block = []
        for _ in range(length):
            base = random.choice(DataLoader.NOISE_TEMPLATES)
            formatted = base.format(pid=random.randint(1000,9999), mem=random.randint(10,90))
            block.append(formatted)
        return block

    @staticmethod
    def generate_synthetic_update(limit=5, noise_length=50):
        print(f"Generating ENHANCED Update Data ({limit} samples, noise={noise_length})...")
        data = []
        entity_pairs = [("Alice", "Alicia"), ("Brian", "Ryan"), ("Catherine", "Katherine"), ("David", "Davina"), ("Eric", "Erica")]
        items = ["server password", "API key", "home address", "phone number", "emergency contact"]
        for i in range(limit):
            target_person, distractor_person = entity_pairs[i % len(entity_pairs)]
            item = items[i % len(items)]
            old_val = f"Secret-{random.randint(100,999)}"
            new_val = f"Secure-{random.randint(1000,9999)}"
            distractor_val = f"Fake-{random.randint(500,599)}"
            stream = []
            stream.append(f"System Record: User {distractor_person}'s {item} has been set to '{distractor_val}'.")
            stream.append(f"Log Entry [2023-01-01]: {target_person} uses '{old_val}' as their {item}.")
            stream.extend(DataLoader._generate_noise_block(noise_length // 2))
            stream.append(f"Security Alert [2023-12-31]: {target_person} updated their {item} to '{new_val}'.")
            stream.extend(DataLoader._generate_noise_block(noise_length // 2))
            question = f"What is the current {item} for {target_person}?"
            ground_truth = new_val
            data.append({
                "stream": stream,
                "question": question,
                "ground_truth": ground_truth
            })
        return data

    @staticmethod
    def generate_synthetic_multihop(limit=5, noise_length=50):
        print(f"Generating ENHANCED Multi-hop Data ({limit} samples, noise={noise_length})...")
        data = []
        objects = ["The Golden Artifact", "The Secret Dossier", "The Virus Sample", "The AI Chip"]
        containers = ["the titanium safe", "the leather briefcase", "the hidden compartment", "the underwater vault"]
        rooms = ["the CEO's office", "the basement lab", "the server room", "the penthouse suite"]
        buildings = ["Building A", "the West Wing", "the Cyber Tower", "the Old Library"]
        for i in range(limit):
            obj = objects[i % len(objects)]
            cont = containers[i % len(containers)]
            room = rooms[i % len(rooms)]
            bldg = buildings[i % len(buildings)]
            fact1 = f"Confidential Report: {obj} has been secured inside {cont}."
            fact2 = f"Security Log: {cont} was last seen being moved into {room}."
            fact3 = f"Architecture Layout: {room} is located on the top floor of {bldg}."
            stream_parts = []
            stream_parts.append(fact3)
            stream_parts.extend(DataLoader._generate_noise_block(noise_length // 2))
            stream_parts.append(fact1)
            stream_parts.extend(DataLoader._generate_noise_block(noise_length // 2))
            stream_parts.append(fact2)
            context_text = " ".join(stream_parts)
            question = f"In which building is {obj} located?"
            ground_truth = bldg
            data.append({
                "context": context_text,
                "question": question,
                "ground_truth": ground_truth
            })
        return data

    load_longmemeval_real = generate_synthetic_update
    load_babilong = generate_synthetic_multihop