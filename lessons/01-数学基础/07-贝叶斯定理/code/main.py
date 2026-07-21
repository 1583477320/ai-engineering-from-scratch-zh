"""朴素贝叶斯分类器——从零实现。"""
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self,smoothing=1.0):
        self.smoothing=smoothing; self.class_counts=defaultdict(int)
        self.word_counts=defaultdict(lambda:defaultdict(int))
        self.class_word_totals=defaultdict(int); self.vocab=set()
    def train(self,docs,labels):
        for doc,label in zip(docs,labels):
            self.class_counts[label]+=1
            for w in doc.lower().split():
                self.word_counts[label][w]+=1; self.class_word_totals[label]+=1; self.vocab.add(w)
    def predict(self,doc):
        words=doc.lower().split(); total=sum(self.class_counts.values()); vs=len(self.vocab)
        best,best_s=None,float("-inf")
        for cls in self.class_counts:
            score=math.log(self.class_counts[cls]/total)
            for w in words:
                score+=math.log((self.word_counts[cls].get(w,0)+self.smoothing)/(self.class_word_totals[cls]+self.smoothing*vs))
            if score>best_s: best_s,best=score,cls
        return best

def main():
    docs=["win free money","free lottery","claim prize","meeting tomorrow","project update","review PR"]
    labels=["spam","spam","spam","ham","ham","ham"]
    clf=NaiveBayes(); clf.train(docs,labels)
    for msg in ["free money waiting","meeting friday","you won prize"]:
        print(f"  '{msg}' → {clf.predict(msg)}")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
