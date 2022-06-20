import qiime2.plugin.model as model


class CaseMatchOneToManyFormat(model.TextFileFormat):
    def validate(self, *args):
        pass


CaseMatchOneToManyDirFmt = model.SingleFileDirectoryFormat(
    "CaseMatchOneToManyFormat",
    "casematch.json",
    CaseMatchOneToManyFormat
)
