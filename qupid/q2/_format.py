import qiime2.plugin.model as model


class CaseMatchFormat(model.TextFileFormat):
    def validate(self, *args):
        pass


class CaseMatchCollectionFormat(model.TextFileFormat):
    def validate(self, *args):
        pass


CaseMatchDirFmt = model.SingleFileDirectoryFormat(
    "CaseMatchFormat",
    "casematch.json",
    CaseMatchFormat
)

CaseMatchCollectionDirFmt = model.SingleFileDirectoryFormat(
    "CaseMatchCollectionFormat",
    "casematch_collection.tsv",
    CaseMatchCollectionFormat
)
