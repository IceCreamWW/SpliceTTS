from .load_audio import load_audio
from .phones2candidates import Phones2Candidates

def get_range_for_rank(n, word_size, rank):
    base, reminder = n // word_size, n % word_size
    start, end = base * rank, base * (rank + 1)
    if rank == word_size - 1:
        end += reminder
    return start, end
