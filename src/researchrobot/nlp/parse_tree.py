import spacy
from nltk import Tree
from spacy.tokens.token import Token

# When walking the tree, exclude this token

try:
    Token.set_extension("prune", default=False)
    Token.set_extension("is_restr", default=None)
except ValueError:
    pass  # already set


class LabelWrapper:
    def __init__(self, token):
        self.tok = token

    def split(self, c):
        return str(self).split(c)

    def __len__(self):
        return len(str(self))

    def __str__(self):
        return self.tok.orth_

    def __repr__(self):
        return str(self)


def nltk_spacy_tree(doc):
    def to_nltk_tree(node):

        if node.n_lefts + node.n_rights > 0:

            c = [to_nltk_tree(child) for child in node.children]

            r = Tree(LabelWrapper(node), c)
        else:
            r = LabelWrapper(node)

        return r

    tree = [to_nltk_tree(sent.root) for sent in doc.sents]

    return tree


def dump_tree(doc):
    return nltk_spacy_tree(doc)[0]


def visit(node, f, memo=None, level=0):
    if memo is None:
        memo = {}

    f(node, memo, level)

    if node.n_lefts + node.n_rights > 0:

        for child in node.children:
            visit(child, f, memo, level + 1)

    return memo


def has_adp(node):
    """Return true if the subtree includes and ADP, excluding the root"""

    return any(n.pos_ == "ADP" for n in list(node.subtree)[1:])


def decomp(doc, f):
    o = {"phrases": [], "base": None}

    s = list(doc.sents)[0].root

    memo = {"phrases": []}

    visit(s, f, memo)

    for i in memo["phrases"]:
        root = doc[i]
        level = memo[i]
        toks = [tok for tok in root.subtree if memo[tok.i] == level]
        o["phrases"].append(toks)
