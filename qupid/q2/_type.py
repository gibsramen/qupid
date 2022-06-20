from qiime2.plugin import SemanticType


CaseMatch = SemanticType("CaseMatch", field_names="mapping")
OneToMany = SemanticType("OneToMany", variant_of=CaseMatch.field["mapping"])
OneToOne = SemanticType("OneToOne", variant_of=CaseMatch.field["mapping"])
