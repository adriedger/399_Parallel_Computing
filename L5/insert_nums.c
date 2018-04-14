#include <stdio.h>
#include <stdlib.h>

struct node {
	int val;
	struct node * left;
	struct node * right;
};

struct node * root = NULL;

int insert_num(int val) {

	struct node * new_node, *parent;
	new_node = malloc(sizeof(new_node));
	new_node->val = val;
	new_node->left = new_node->right = NULL;

	if (root == NULL) {
		root = new_node;
		return 0;
	}

#ifdef LINKED
	if (new_node->val < root->val) {
		new_node->left = root;	
		root = new_node;
		return 0;
	}

	parent = root;
	while ((parent->left != NULL) && (new_node->val > parent->left->val)) {
		parent = parent->left;
	}
	new_node->left = parent->left;
	parent->left = new_node;
	return 0;
	
#else
	parent = root;
	while (1) {
		if (new_node->val < parent->val) {
			if (parent->left == NULL) {
				parent->left = new_node;
				return 0;
			} else {
			   	parent = parent->left;
			}
		} else if (parent->right == NULL) {
				parent->right = new_node;
				return 0;
			} else {
			   	parent = parent->right;
			}
	}
#endif

}

void print_struct(struct node * root) {

#ifdef LINKED
	while (root != NULL) {
		printf("%14d\n", root->val);
		root=root->left;
	}	
#else

	if (root == NULL)
		return;
	print_struct(root->left);
	printf("%14d\n", root->val);
	print_struct(root->right);
#endif
}

int main(int argc, char ** argv) {

	int count = atoi(argv[1]);

	// uncomment the commented lines below for some output	
	srandom(0);
	for (int i = 0; i < count; i++) {
		int x = random();
//`		printf("%14d\n", x);
		insert_num(x);
	}

//	printf("and the numbers...\n");
//	print_struct(root);
}
