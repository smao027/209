class snp(object):
    def __init__(self, snp_id, chromosome, position, allele1, allele2):
        self.snp_id = snp_id
        self.chromosome = chromosome
        self.position = position
        self.allele1 = allele1
        self.allele2 = allele2

    def __repr__(self):
        return f"snp({self.snp_id}, {self.chromosome}, {self.position}, {self.allele1}, {self.allele2})"
    def __str__(self):
        return f"SNP ID: {self.snp_id}, Chromosome: {self.chromosome}, Position: {self.position}, Allele1: {self.allele1}, Allele2: {self.allele2}"
    def __eq__(self, other):
        if not isinstance(other, snp):
            return NotImplemented
        return (self.snp_id == other.snp_id and
                self.chromosome == other.chromosome and
                self.position == other.position and
                self.allele1 == other.allele1 and
                self.allele2 == other.allele2)
    def __hash__(self):
        return hash((self.snp_id, self.chromosome, self.position, self.allele1, self.allele2))
    def __lt__(self, other):
        if not isinstance(other, snp):
            return NotImplemented
        return (self.chromosome, self.position) < (other.chromosome, other.position)
    def __le__(self, other):
        if not isinstance(other, snp):
            return NotImplemented
        return (self.chromosome, self.position) <= (other.chromosome, other.position)
    def __gt__(self, other):
        if not isinstance(other, snp):
            return NotImplemented
        return (self.chromosome, self.position) > (other.chromosome, other.position)
    def __ge__(self, other):