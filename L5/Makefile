CFLAGS=-D_BSD_SOURCE -std=c99 -g

all: linked_nums tree_nums matmulti

linked_nums: insert_nums.c
	$(CC) $(CFLAGS) $^ -o $@ -DLINKED

tree_nums: insert_nums.c
	$(CC) $(CFLAGS) $^ -o $@

matmulti: matmulti.c
	$(CC) $^ -o $@ -lpthread -g

clean:
	rm -f tree_nums linked_nums
