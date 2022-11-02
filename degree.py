from torch_geometric.utils import degree

ROOT = "/dbfs/mnt/ogb2022"
dataset = MAG240MDataset(root=ROOT)


def get_in_and_out_degree(edge_index, src_nodes, dst_nodes):
    out_degree = degree(edge_index[0], src_nodes, dtype=torch.float16)
    in_degree = degree(edge_index[1], dst_nodes, dtype=torch.float16)

    return out_degree, in_degree


if __name__ == "__main__":

    paper_cites_paper = torch.from_numpy(dataset.edge_index('paper', 'paper'))
    paper_out_citations, paper_in_citations = get_in_and_out_degree(
        paper_cites_paper, dataset.num_papers, dataset.num_papers)

    paper_cites_paper = torch.from_numpy(dataset.edge_index('author', 'paper'))
    author_out, paper_in_writes = get_in_and_out_degree(
        paper_cites_paper, dataset.num_authors, dataset.num_papers)

    paper_cites_paper = torch.from_numpy(dataset.edge_index('author', 'institution')
                                         )
    author_affiliations, institute_authors = get_in_and_out_degree(
        paper_cites_paper, dataset.num_authors, dataset.num_institutions)

    paper_degree_features = torch.cat(
        [paper_out_citations.reshape(-1, 1),
         paper_in_citations.reshape(-1, 1),
         paper_in_writes.reshape(-1, 1)],
        dim=1)

    author_degree_features = torch.cat(
        [author_out.reshape(-1, 1),
         author_affiliations.reshape(-1, 1),
         torch.zeros((author_affiliations.shape[0], 1), dtype=torch.float16)],
        dim=1)

    institution_degree_features = torch.cat(
        [institute_authors.reshape(-1, 1),
         torch.zeros((institute_authors.shape[0], 2), dtype=torch.float16)
         ],
        dim=1)

    x_degree_features = torch.cat(
        [paper_degree_features, author_degree_features, institution_degree_features], dim=0)

    torch.save(x_degree_features,
               '/dbfs/mnt/ogb2022/mag240m_kddcup2021/whitening_128/pai_degree_information.pt')
