
class CoresetTreeServiceUnsupervisedMixin:
    pass

class CoresetTreeServiceSupervisedMixin:
    pass

    def build(self, X: Union[Iterable, Iterable[Iterable]], y: Union[Iterable[Any], Iterable[Iterable[Any]]], indices: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, props: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Create a coreset tree from the parameters X, y, indices and props (properties).
        build functions may be called only once. To add more data to the coreset tree use one of
        the partial_build functions. All features must be numeric.

        Parameters:
            X: array like or iterator of arrays like.
                An array or an iterator of features. All features must be numeric.

            y: array like or iterator of arrays like.
                An array or an iterator of targets.

            indices: array like or iterator of arrays like, optional.
                An array or an iterator with indices of X.

            props: array like or iterator of arrays like, optional.
                An array or an iterator of properties.
                Properties, won’t be used to compute the Coreset or train the model, but it is possible to
                filter_out_samples on them or to pass them in the select_from_function of get_important_samples.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False.
                False (default) - input data might be updated as result a consequence action like update_targets or update_features
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

    def partial_build(self, X: Union[Iterable, Iterable[Iterable]], y: Union[Iterable[Any], Iterable[Iterable[Any]]], indices: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, props: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Add new samples to a coreset tree from parameters X, y, indices and props (properties).
        All features must be numeric.

        Parameters:
            X: array like or iterator of arrays like.
                An array or an iterator of features. All features must be numeric.

            y: array like or iterator of arrays like.
                An array or an iterator of targets.

            indices: array like or iterator of arrays like, optional.
                An array or an iterator with indices of X.

            props: array like or iterator of arrays like, optional.
                An array or an iterator of properties.
                Properties, won’t be used to compute the Coreset or train the model, but it is possible to
                filter_out_samples on them or to pass them in the select_from_function of get_important_samples.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False.
                False (default) - input data might be updated as result a consequence action like update_targets or update_features.
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

class CoresetTreeServiceClassifierMixin:
    pass

    def get_coreset(self, level: int=0, as_orig: bool=False, with_index: bool=False, inverse_class_weight: bool=True) -> dict:
        """
        Get tree's coreset data either in a processed format or in the original format.
        Use the level parameter to control the level of the tree from which samples will be returned.

        Parameters:
            level: int, optional, default 0.
                Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2*coreset_size samples, etc.

            as_orig: boolean, optional, default False.
                Should the data be returned in its original format or as a tuple of indices, X, and optionally y.
                True: data is returned as a pandas DataFrame.
                False: return a tuple of (indices, X, y) if target was used and (indices, X) when there is no target.

            with_index: boolean, optional, default False.
                Relevant only when as_orig=True. Should the returned data include the index column.

            inverse_class_weight: boolean, default True.
                True - return weights / class_weights.
                False - return weights as they are.
                Relevant only for classification tasks and only if class_weight was passed in
                the coreset_params when initializing the class.

        Returns:
            Dict:
                data: numpy arrays tuple (indices, X, optional y) or a pandas DataFrame.
                w: A numpy array of sample weights.
                n_represents: number of instances represented by the coreset."""
        pass

    def get_important_samples(self, size: int=None, class_size: Dict[Any, Union[int, str]]=None, ignore_indices: Iterable=None, select_from_indices: Iterable=None, select_from_function: Callable[[Iterable, Iterable, Union[Iterable, None], Union[Iterable, None]], Iterable[Any]]=None, ignore_seen_samples: bool=True) -> Union[ValueError, dict]:
        '''
        Returns indices of samples in descending order of importance. Useful for identifying mislabeled instances.
        Either class_size (recommended) or size must be provided. Must be called after build.

        Parameters:
            size: int, optional
                Number of samples to return.
                When class_size is provided, remaining samples are taken from classes not appearing in class_size dictionary.

            class_size: dict {class: int or "all" or "any"}, optional.
                Controls the number of samples to choose for each class.
                int: return at most size.
                "all": return all samples.
                "any": limits the returned samples to the specified classes.

            ignore_indices: array-like, optional.
                An array of indices to ignore when selecting important samples.

            select_from_indices: array-like, optional.
                 An array of indices to consider when selecting important samples.

            select_from_function: function, optional.
                 Pass a function in order to limit the selection of the important samples accordingly.
                 The function should accept 4 parameters as input: indices, X, y, properties
                 and return a list(iterator) of the desired indices.

            ignore_seen_samples: bool, optional, default True.
                 Exclude already seen samples and set the seen flag on any indices returned by the function.

        Returns:
            Dict:
                indices: array-like[int].
                    Important samples indices.
                X: array-like[int].
                    X array.
                y: array-like[int].
                    y array.
                importance: array-like[float].
                    The importance property. Instances that receive a high Importance in the Coreset computation,
                    require attention as they usually indicate a labeling error,
                    anomaly, out-of-distribution problem or other data-related issue.

        Examples
        -------
        Input:
            size=100,
            class_size={"class A": 10, "class B": 50, "class C": "all"}
        Output:
            10 of "class A",
            50 of "class B",
            12 of "class C" (all),
            28 of "class D/E"'''
        pass
