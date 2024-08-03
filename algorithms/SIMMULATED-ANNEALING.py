from algorithms.utils import get_edges, plot_edges

TYPES = ['packing', 'separated']
FILES = ['instance_01_2pol', 'instance_01_3pol', 'instance_01_4pol', 'instance_01_5pol', 'spfc_instance']

if __name__ == '__main__':
    for instance_type in TYPES:
        for instance_file in FILES:
            instance = f"ejor/{instance_type}/{instance_file}.txt"
            edges = get_edges(file_name=instance)
            # plot_edges(title=f"{instance_type}/{instance_file}", edges=edges)
